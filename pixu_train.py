import argparse
import os
import os.path as path
import random
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
import csv

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Global device setup for multiple GPUs
if torch.cuda.device_count() > 1:
    device = torch.device("cuda")  # Use all available GPUs
    print(f"Using {torch.cuda.device_count()} GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

image_data_path = 'pixu/images'

# Clear GPU memory
torch.cuda.empty_cache()

def validate_tsv_file(filepath, column_name, expected_count=3):
    """Validate that a TSV column contains space-separated strings with expected count."""
    try:
        df = pd.read_table(filepath, dtype={column_name: str})
        invalid_rows = df[~df[column_name].apply(
            lambda x: isinstance(x, str) and len(x.strip().split()) == expected_count and not any(c in x for c in ['(', ')', '[', ']', ','])
        )]
        if not invalid_rows.empty:
            print(f"Warning: Found {len(invalid_rows)} invalid rows in {filepath} for column {column_name}")
    except Exception as e:
        print(f"Error validating {filepath}: {str(e)}")

class ConfigManager:
    model_name = 'PIXU'

    def __init__(self):
        parser = argparse.ArgumentParser(description="Configuration settings for the PIXU model")
        parser.add_argument('--num_categories', type=int, default=274 + 1)
        parser.add_argument('--image_embedding_dim', type=int, default=768)
        parser.add_argument('--data_dir', type=str, default='pixu/preprocessed_data', help='Path to preprocessed data')
        parser.add_argument('--algorithm', type=str, default='core', choices=['core', 'visual', 'category', 'temporal', 'personalized'], help='Algorithm for data')
        parser.add_argument('--current_data_path', type=str, default='pixu/trained_model')
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--dropout_probability', type=float, default=0.2)
        parser.add_argument('--num_attention_heads', type=int, default=16)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--pretrained_model_name', type=str, default='openai/clip-vit-base-patch32')
        args = parser.parse_args()
        self.args = args

    class BaseConfig:
        def __init__(self, args):
            self.pretrained_model_name = args.pretrained_model_name
            self.original_data_path = args.data_dir
            self.current_data_path = args.current_data_path
            self.algorithm = args.algorithm
            self.num_epochs = 10
            self.num_batches_show_loss = 100
            self.num_batches_validate = 200
            self.batch_size = args.batch_size
            self.learning_rate = args.learning_rate
            self.num_workers = 2
            self.num_clicked_news_a_user = 50
            self.dropout_probability = args.dropout_probability
            self.num_users = 50000 + 1
            self.image_embedding_dim = args.image_embedding_dim
            self.query_vector_dim = 200

    class PIXUConfig(BaseConfig):
        def __init__(self, args):
            super().__init__(args)
            self.dataset_attributes = {"news": ['image'], "record": []}
            self.num_attention_heads = args.num_attention_heads

    def get_config(self):
        return self.PIXUConfig(self.args)

def custom_collate_fn(batch):
    users = [item['user'] for item in batch]
    news_ids = [item['news_id'] for item in batch]
    clicked = [item['clicked'] for item in batch]
    candidate_images = [item['candidate_images'] for item in batch]
    clicked_news = [item['clicked_news'] for item in batch]
    clicked_news_lengths = torch.tensor([item['clicked_news_length'] for item in batch])

    candidate_images_tensor = torch.stack([torch.stack(cn) for cn in candidate_images])
    max_clicked_len = max(len(cn) for cn in clicked_news)
    padded_clicked_news = []
    for cn in clicked_news:
        padding = [torch.zeros(3, 224, 224)] * (max_clicked_len - len(cn))
        padded_clicked_news.append(cn + padding)
    clicked_news_tensor = torch.stack([torch.stack(cn) for cn in padded_clicked_news])

    return {
        'user': torch.tensor(users),
        'news_id': news_ids,
        'clicked': clicked,
        'candidate_images': candidate_images_tensor,
        'clicked_news': clicked_news_tensor,
        'clicked_news_length': clicked_news_lengths
    }

def user_collate_fn(batch):
    users = [item['user'] for item in batch]
    clicked_news_strings = [item['clicked_news_string'] for item in batch]
    clicked_news = [item['clicked_news'] for item in batch]
    clicked_news_lengths = torch.tensor([item['clicked_news_length'] for item in batch])

    max_clicked_len = max(len(cn) for cn in clicked_news)
    padded_clicked_news = []
    for cn in clicked_news:
        padding = [''] * (max_clicked_len - len(cn))
        padded_clicked_news.append(cn + padding)
    
    return {
        'user': torch.tensor(users),
        'clicked_news_string': clicked_news_strings,
        'clicked_news': padded_clicked_news,
        'clicked_news_length': clicked_news_lengths
    }

class DatasetManager:
    class BaseDataset(Dataset):
        def __init__(self, behaviors_path, news_path, config):
            super().__init__()
            self.config = config
            self.behaviors_parsed = pd.read_table(behaviors_path)
            self.news_parsed = pd.read_table(news_path, index_col='id', usecols=['id', 'image', 'category'], dtype={'id': str})
            self.news_id2int = {x: i for i, x in enumerate(self.news_parsed.index)}
            self.news2dict = self.news_parsed.to_dict('index')
            self.image_dir = image_data_path
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26130258, 0.26130258, 0.27577711])
            ])

        def __len__(self):
            return len(self.behaviors_parsed)

        def __getitem__(self, idx):
            item = {}
            row = self.behaviors_parsed.iloc[idx]
            item['user'] = row.user
            item['news_id'] = row.news_id
            clicked = row.clicked.strip().split()
            if len(clicked) != 3:
                print(f"Warning: Invalid clicked length at index {idx}: {clicked}")
                clicked = ['1', '0', '0']
            item['clicked'] = [int(x) for x in clicked]
            candidate_images = row.candidate_images.strip().split()
            if len(candidate_images) != 3:
                print(f"Warning: Invalid candidate_images length at index {idx}: {candidate_images}")
                candidate_images = [row.news_id, row.news_id, row.news_id]
            item['candidate_images'] = [self._load_news_image(x) for x in candidate_images]
            clicked_news = [self._load_news_image(x) for x in row.clicked_news.split()[:self.config.num_clicked_news_a_user]]
            item['clicked_news'] = clicked_news
            item['clicked_news_length'] = len(clicked_news)
            return item

        def _load_news_image(self, news_id):
            img_path = path.join(self.image_dir, f"{news_id}.jpg")
            try:
                img = Image.open(img_path).convert('RGB')
                return self.transform(img)
            except Exception as e:
                print(f"Warning: Failed to load image {img_path}: {str(e)}")
                return torch.zeros(3, 224, 224)

    class NewsDataset(Dataset):
        def __init__(self, news_path, config):
            super().__init__()
            self.config = config
            self.news_parsed = pd.read_table(news_path, usecols=['id', 'image', 'category'], dtype={'id': str})
            self.image_dir = image_data_path
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26130258, 0.26130258, 0.27577711])
            ])
            self.news_ids = self.news_parsed['id'].tolist()
            self.news2dict = {row['id']: row for _, row in self.news_parsed.iterrows()}

        def __len__(self):
            return len(self.news_ids)

        def __getitem__(self, idx):
            news_id = self.news_ids[idx]
            img_path = path.join(self.image_dir, f"{news_id}.jpg")
            try:
                img = Image.open(img_path).convert('RGB')
                tensor = self.transform(img)
                return {"id": news_id, "image": tensor, "category": self.news2dict[news_id]['category']}
            except Exception as e:
                tensor = torch.zeros(3, 224, 224)
                return {"id": news_id, "image": tensor, "category": self.news2dict[news_id]['category']}

    class UserDataset(Dataset):
        def __init__(self, behaviors_path, user2int_path, config):
            super().__init__()
            self.config = config
            self.behaviors = pd.read_table(behaviors_path, header=None, usecols=[1, 3], names=['user', 'clicked_news'])
            self.behaviors['clicked_news'] = self.behaviors['clicked_news'].fillna(' ')
            self.behaviors.drop_duplicates(inplace=True)
            user2int = dict(pd.read_table(user2int_path).values.tolist())
            for row in tqdm(self.behaviors.itertuples(), total=self.behaviors.shape[0], desc="Processing User Dataset"):
                if row.user in user2int:
                    self.behaviors.at[row.Index, 'user'] = user2int[row.user]
                else:
                    self.behaviors.at[row.Index, 'user'] = 0

        def __len__(self):
            return len(self.behaviors)

        def __getitem__(self, idx):
            row = self.behaviors.iloc[idx]
            item = {
                "user": row.user,
                "clicked_news_string": row.clicked_news,
                "clicked_news": row.clicked_news.split()[:self.config.num_clicked_news_a_user],
                "clicked_news_length": len(row.clicked_news.split()[:self.config.num_clicked_news_a_user])
            }
            return item

    class ParsedBehaviorsDataset(Dataset):
        def __init__(self, behaviors_path):
            super().__init__()
            self.behaviors = pd.read_table(behaviors_path, dtype={'candidate_images': str})
            self.behaviors['clicked_news'] = self.behaviors['clicked_news'].fillna(' ')
            self.behaviors['candidate_images'] = self.behaviors['candidate_images'].astype(str).str.strip('()').str.replace("'", "").str.replace(",", "")

            def parse_candidate_images(x):
                x = str(x).strip('()').replace("'", "").replace(",", "").replace("[", "").replace("]", "")
                split_ids = x.strip().split()
                if len(split_ids) != 3:
                    print(f"Warning: Parsed invalid candidate_images: {x}, expected 3 IDs")
                    if split_ids and split_ids[0] != '':
                        return [split_ids[0]] * 3
                    return ['dummy'] * 3
                return split_ids

            self.behaviors['candidate_images'] = self.behaviors['candidate_images'].apply(parse_candidate_images)
            self.behaviors['clicked'] = self.behaviors['clicked'].apply(
                lambda x: x.strip().split() if isinstance(x, str) else ['1', '0', '0']
            )
            valid_rows = []
            invalid_count = 0
            for idx, row in self.behaviors.iterrows():
                candidate_images = row['candidate_images']
                clicked = row['clicked']
                if len(candidate_images) != 3 or any(id == 'dummy' for id in candidate_images):
                    print(f"Warning: Invalid candidate_images at index {idx}: {candidate_images}")
                    invalid_count += 1
                    continue
                if len(clicked) != 3 or sum(int(x) for x in clicked) != 1:
                    print(f"Warning: Invalid clicked labels at index {idx}: {clicked}")
                    invalid_count += 1
                    continue
                valid_rows.append(row)
            self.behaviors = pd.DataFrame(valid_rows)
            print(f"Number of invalid rows skipped: {invalid_count}")

        def __len__(self):
            return len(self.behaviors)

        def __getitem__(self, idx):
            row = self.behaviors.iloc[idx]
            clicked = row.clicked
            candidate_images = row.candidate_images
            return {
                "user": row.user,
                "news_id": row.news_id,
                "clicked_news_string": row.clicked_news,
                "candidate_images": candidate_images,
                "clicked": [int(x) for x in clicked]
            }

    class BehaviorsDataset(Dataset):
        def __init__(self, behaviors_path):
            super().__init__()
            self.behaviors = pd.read_table(behaviors_path, header=None, usecols=range(5), names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])
            self.behaviors['clicked_news'] = self.behaviors['clicked_news'].fillna(' ')
            self.behaviors.impressions = self.behaviors.impressions.str.split()

        def __len__(self):
            return len(self.behaviors)

        def __getitem__(self, idx):
            row = self.behaviors.iloc[idx]
            return {
                "impression_id": row.impression_id,
                "user": row.user,
                "time": row.time,
                "clicked_news_string": row.clicked_news,
                "impressions": row.impressions
            }

class Model:
    class ModelInitializer:
        def init_weights(self, m: nn.Module):
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight.data)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    class Attention:
        class ScaledDotProductAttention(nn.Module):
            def __init__(self, d_k):
                super().__init__()
                self.d_k = d_k

            def forward(self, Q, K, V, attn_mask=None):
                scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
                if attn_mask is not None:
                    scores = scores + attn_mask
                attn = F.softmax(scores, dim=-1)
                context = torch.matmul(attn, V)
                return context, attn

        class ALiBiMultiHeadSelfAttention(nn.Module):
            def __init__(self, d_model, num_attention_heads):
                super().__init__()
                self.d_model = d_model
                self.num_attention_heads = num_attention_heads
                self.d_k = d_model // num_attention_heads
                self.d_v = d_model // num_attention_heads
                self.W_Q = nn.Linear(d_model, d_model)
                self.W_K = nn.Linear(d_model, d_model)
                self.W_V = nn.Linear(d_model, d_model)
                self._initialize_weights()
                slopes = [2 ** (-2 ** -(np.log2(num_attention_heads) - i)) for i in range(num_attention_heads)]
                self.register_buffer('slopes', torch.tensor(slopes, dtype=torch.float32))

            def _initialize_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight, gain=1)

            def forward(self, Q, K=None, V=None, length=None):
                if K is None: K = Q
                if V is None: V = Q
                batch_size = Q.size(0)
                seq_len = Q.size(1)
                q_s = self.W_Q(Q).view(batch_size, -1, self.num_attention_heads, self.d_k).transpose(1, 2)
                k_s = self.W_K(K).view(batch_size, -1, self.num_attention_heads, self.d_k).transpose(1, 2)
                v_s = self.W_V(V).view(batch_size, -1, self.num_attention_heads, self.d_v).transpose(1, 2)

                positions = torch.arange(seq_len, device=Q.device).unsqueeze(0).expand(batch_size, seq_len)
                relative_positions = positions.unsqueeze(-1) - positions.unsqueeze(-2)
                alibi_bias = relative_positions.unsqueeze(1) * self.slopes.view(1, self.num_attention_heads, 1, 1)
                alibi_bias = alibi_bias * -1

                context, _ = Model.Attention.ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s, attn_mask=alibi_bias)
                context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_attention_heads * self.d_v)
                return context

        class AdditiveAttention(nn.Module):
            def __init__(self, query_vector_dim, candidate_vector_dim):
                super().__init__()
                self.linear = nn.Linear(candidate_vector_dim, query_vector_dim)
                self.attention_query_vector = nn.Parameter(torch.empty(query_vector_dim).uniform_(-0.1, 0.1))

            def forward(self, candidate_vector):
                temp = torch.tanh(self.linear(candidate_vector))
                candidate_weights = F.softmax(torch.matmul(temp, self.attention_query_vector), dim=1)
                target = torch.bmm(candidate_weights.unsqueeze(dim=1), candidate_vector).squeeze(dim=1)
                return target

    class NewsEncoder(nn.Module, ModelInitializer):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.clip_model = CLIPModel.from_pretrained(config.pretrained_model_name).vision_model
            self.processor = CLIPProcessor.from_pretrained(config.pretrained_model_name, use_fast=True)
            self.dim = 768
            for param in self.clip_model.parameters():
                param.requires_grad = False
            for param in self.clip_model.encoder.layers[-5:].parameters():
                param.requires_grad = True
            for param in self.clip_model.post_layernorm.parameters():
                param.requires_grad = True
            self.pooler = nn.Sequential(
                nn.Linear(self.dim, self.dim),
                nn.Dropout(0.1),
                nn.LayerNorm(self.dim),
                nn.SiLU(),
            )
            self.pooler.apply(self.init_weights)
            self.multihead_self_attention = Model.Attention.ALiBiMultiHeadSelfAttention(self.dim, config.num_attention_heads)
            self.additive_attention = Model.Attention.AdditiveAttention(config.query_vector_dim, self.dim)

        def forward(self, images):
            images = images.to(device)
            batch_size, num_items, c, h, w = images.shape
            images = images.view(-1, c, h, w)
            image_features = self.clip_model(pixel_values=images).last_hidden_state
            image_features = image_features[:, 0, :]
            image_vector = self.pooler(image_features)
            image_vector = image_vector.view(batch_size, num_items, -1)
            multihead_vector = self.multihead_self_attention(image_vector)
            multihead_vector = F.dropout(multihead_vector, p=self.config.dropout_probability, training=self.training)
            final_vector = self.additive_attention(multihead_vector)
            return final_vector

    class UserEncoder(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.multihead_self_attention = Model.Attention.ALiBiMultiHeadSelfAttention(config.image_embedding_dim, config.num_attention_heads)
            self.additive_attention = Model.Attention.AdditiveAttention(config.query_vector_dim, config.image_embedding_dim)

        def forward(self, user_vector, lengths):
            multihead_user_vector = self.multihead_self_attention(user_vector, length=lengths)
            multihead_user_vector = F.dropout(multihead_user_vector, p=self.config.dropout_probability, training=self.training)
            final_user_vector = self.additive_attention(multihead_user_vector)
            return final_user_vector

    class DotProductClickPredictor(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, candidate_image_vectors, user_vector):
            user_vector = user_vector.unsqueeze(-1)
            scores = torch.bmm(candidate_image_vectors, user_vector).squeeze(-1)
            return scores

    class PIXU(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.news_encoder = Model.NewsEncoder(config)
            self.user_encoder = Model.UserEncoder(config)
            self.click_predictor = Model.DotProductClickPredictor()

        def forward(self, candidate_images, clicked_news, clicked_news_lengths):
            batch_size, num_candidates, c, h, w = candidate_images.shape
            candidate_images = candidate_images.view(-1, c, h, w)
            candidate_image_vector = self.news_encoder(candidate_images.unsqueeze(1))
            candidate_image_vector = candidate_image_vector.view(batch_size, num_candidates, -1)

            clicked_news_vector = self.news_encoder(clicked_news)
            user_vector = self.user_encoder(clicked_news_vector.unsqueeze(1), clicked_news_lengths)
            click_probability = self.click_predictor(candidate_image_vector, user_vector)
            return click_probability

        def get_news_vector(self, news):
            images = news["image"]
            if len(images.shape) == 3:
                images = images.unsqueeze(0)
            images = images.unsqueeze(1)
            vectors = self.news_encoder(images)
            return vectors

        def get_user_vector(self, clicked_news_vector, lengths):
            return self.user_encoder(clicked_news_vector, lengths)

        def get_prediction(self, image_vector, user_vector):
            return self.click_predictor(image_vector.unsqueeze(dim=0), user_vector.unsqueeze(dim=0)).squeeze(dim=0)

class MetricsEvaluator:
    def calculate_single_user_metric(self, pair):
        y_true, y_score = pair
        try:
            y_true = [int(x) for x in y_true]
            y_score = [float(x) for x in y_score]
            if len(y_true) != 3 or len(y_score) != 3:
                print(f"Warning: Invalid task lengths - y_true: {len(y_true)}, y_score: {len(y_score)}")
                return [np.nan] * 4
            auc = roc_auc_score(y_true, y_score)
            y_pred = (np.array(y_score) >= np.max(y_score)).astype(int)
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            return [auc, accuracy, precision, recall]
        except Exception as e:
            print(f"Error in calculate_single_user_metric: {str(e)}")
            return [np.nan] * 4

    @torch.no_grad()
    def evaluate(self, model, directory, num_workers, news_dataset_built=None, max_count=sys.maxsize):
        dataset_manager = DatasetManager()
        config = model.module.config  # Access config from the underlying model when using DataParallel

        if news_dataset_built:
            news_dataset = news_dataset_built
        else:
            news_dataset = dataset_manager.NewsDataset(path.join(directory, 'dev_news_parsed.tsv'), config)

        news_dataloader = DataLoader(news_dataset, batch_size=config.batch_size * 16, shuffle=False, num_workers=0, drop_last=False, pin_memory=False)

        news2vector = {}
        for minibatch in news_dataloader:
            news_ids = minibatch["id"]
            news_vectors = model.module.get_news_vector(minibatch)  # Access module for DataParallel
            for id, vector in zip(news_ids, news_vectors):
                if id not in news2vector:
                    news2vector[id] = vector.cpu()

        user_dataset = dataset_manager.UserDataset(path.join(directory, f'behavior_dev_{config.algorithm}.tsv'), path.join(directory, f'user2int_dev_{config.algorithm}.tsv'), config)
        user_dataloader = DataLoader(user_dataset, batch_size=config.batch_size * 16, shuffle=False, num_workers=0, drop_last=False, pin_memory=False, collate_fn=user_collate_fn)

        user2vector = {}
        for minibatch in user_dataloader:
            user_strings = minibatch["clicked_news_string"]
            clicked_news_vectors = []
            lengths = []
            for news_list, length in zip(minibatch["clicked_news"], minibatch["clicked_news_length"]):
                vectors = []
                for x in news_list:
                    if x in news2vector and x != '':
                        vector = news2vector[x].to(device)
                        if vector.shape != torch.Size([config.image_embedding_dim]):
                            vector = torch.zeros(config.image_embedding_dim, device=device)
                        vectors.append(vector)
                if vectors:
                    while len(vectors) < config.num_clicked_news_a_user:
                        vectors.append(torch.zeros(config.image_embedding_dim).to(device))
                    vector_tensor = torch.stack(vectors, dim=0)
                    clicked_news_vectors.append(vector_tensor)
                    lengths.append(len(vectors))
                else:
                    vector_tensor = torch.zeros(config.num_clicked_news_a_user, config.image_embedding_dim).to(device)
                    clicked_news_vectors.append(vector_tensor)
                    lengths.append(config.num_clicked_news_a_user)
            clicked_news_vector = torch.nn.utils.rnn.pad_sequence(clicked_news_vectors, batch_first=True)
            lengths = torch.tensor(lengths).to(device)
            user_vector = model.module.get_user_vector(clicked_news_vector, lengths)  # Access module for DataParallel
            for user, vector in zip(user_strings, user_vector):
                if user not in user2vector:
                    user2vector[user] = vector.cpu()

        behaviors_dataset = dataset_manager.BehaviorsDataset(path.join(directory, f'behavior_dev_{config.algorithm}.tsv'))
        behaviors_dataloader = DataLoader(behaviors_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False, pin_memory=False)

        tasks = []
        count = 0
        for minibatch in behaviors_dataloader:
            count += 1
            if count == max_count:
                break
            impressions = minibatch['impressions']
            if not isinstance(impressions, list) or len(impressions) != 3:
                print(f"Warning: Skipping invalid impressions at count {count}")
                continue
            try:
                candidate_images = []
                y_true = []
                for news in impressions:
                    if not news or not news[0]:
                        raise ValueError("Empty news tuple")
                    parts = news[0].split('-')
                    if len(parts) != 2:
                        raise ValueError("Invalid format")
                    candidate_images.append(parts[0])
                    y_true.append(int(parts[1]))
            except (IndexError, ValueError):
                print(f"Warning: Malformed impressions at count {count}")
                continue
            if len(candidate_images) != 3 or len(y_true) != 3 or sum(y_true) != 1:
                print(f"Warning: Invalid candidate_images or y_true at count {count}")
                continue
            if not all(img in news2vector for img in candidate_images):
                print(f"Warning: Missing news IDs in news2vector at count {count}")
                continue
            if minibatch['clicked_news_string'][0] not in user2vector:
                print(f"Warning: Skipping missing user vector at count {count}")
                continue
            candidate_image_vector = torch.stack([news2vector[img].to(device) for img in candidate_images], dim=0)
            if candidate_image_vector.shape != torch.Size([3, config.image_embedding_dim]):
                print(f"Warning: Skipping invalid candidate_image_vector shape at count {count}")
                continue
            user_vector = user2vector[minibatch['clicked_news_string'][0]].to(device)
            try:
                click_probability = model.module.get_prediction(candidate_image_vector, user_vector)  # Access module for DataParallel
                y_pred = click_probability.cpu().tolist()
            except Exception as e:
                print(f"Error in get_prediction at count {count}: {str(e)}")
                continue
            if len(y_pred) != 3:
                print(f"Warning: Skipping invalid y_pred at count {count}")
                continue
            tasks.append((y_true, y_pred))

        if not tasks:
            print("Error: No valid tasks generated. Check behavior_dev.tsv for dev dataset.")
            return np.nan, np.nan, np.nan, np.nan

        try:
            with Pool(processes=num_workers) as pool:
                results = list(pool.imap(self.calculate_single_user_metric, tasks))
        except Exception as e:
            print(f"Multiprocessing failed: {str(e)}")
            results = [self.calculate_single_user_metric(task) for task in tasks]

        aucs, accuracies, precisions, recalls = np.array(results).T
        return np.nanmean(aucs), np.nanmean(accuracies), np.nanmean(precisions), np.nanmean(recalls)

class Trainer:
    def __init__(self):
        self.model_name = 'PIXU'

    def time_since(self, since):
        now = time.time()
        elapsed_time = now - since
        return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    def train(self):
        config_manager = ConfigManager()
        config = config_manager.get_config()
        dataset_manager = DatasetManager()
        metrics_evaluator = MetricsEvaluator()

        start_time = time.time()
        writer = SummaryWriter(f"pixu/runs/PIXU/{config.algorithm}/{time.strftime('%Y-%m-%d_%H-%M-%S')}")
        checkpoint_dir = path.join(config.current_data_path, 'checkpoint', 'clip', f"{self.model_name}_{config.algorithm}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        torch.cuda.empty_cache()

        model = Model.PIXU(config)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)  # Wrap model with DataParallel for multi-GPU
        model = model.to(device)

        print(model)

        dataset = dataset_manager.BaseDataset(
            path.join(config.original_data_path, f'behaviors_parsed_train_{config.algorithm}.tsv'),
            path.join(config.original_data_path, 'train_news_parsed.tsv'),
            config
        )
        print(f"Loaded dataset with size {len(dataset)}")

        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, drop_last=True, pin_memory=True, collate_fn=custom_collate_fn)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

        evaluate_dataset = dataset_manager.NewsDataset(
            path.join(config.original_data_path, 'dev_news_parsed.tsv'),
            config
        )

        loss_full = []
        step = 0
        best_auc = 0
        best_checkpoint_path = None
        patience = 5
        patience_counter = 0

        print("Training Epochs")
        for epoch in range(config.num_epochs):
            model.train()
            for i, minibatch in enumerate(dataloader):
                step += 1
                clicked_news_lengths = minibatch["clicked_news_length"].to(device)
                candidate_images = minibatch["candidate_images"].to(device)
                clicked_news = minibatch["clicked_news"].to(device)
                y_pred = model(candidate_images, clicked_news, clicked_news_lengths)
                
                batch_size = y_pred.shape[0]
                clicked_labels = torch.tensor([labels.index(1) for labels in minibatch["clicked"]], dtype=torch.long, device=device)
                loss = criterion(y_pred, clicked_labels)
                loss_full.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % config.num_batches_show_loss == 0:
                    print(f"Time {self.time_since(start_time)}, Step {step}, Loss: {loss.item():.4f}, Avg Loss: {np.mean(loss_full):.4f}")

                if step % config.num_batches_validate == 0:
                    model.eval()
                    auc, accuracy, precision, recall = metrics_evaluator.evaluate(model, config.original_data_path, config.num_workers, evaluate_dataset)
                    scheduler.step(auc)
                    print(f"Time {self.time_since(start_time)}, Step {step}, AUC: {auc:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, LR: {optimizer.param_groups[0]['lr']}")
                    if auc > best_auc:
                        best_auc = auc
                        patience_counter = 0
                        if best_checkpoint_path:
                            os.remove(best_checkpoint_path)
                        best_checkpoint_path = path.join(checkpoint_dir, f"ckpt-step-{step}-auc-{auc:.4f}-acc-{accuracy:.4f}-prec-{precision:.4f}-rec-{recall:.4f}.pth")
                        torch.save({
                            'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'step': step,
                            'auc': auc,
                            'epoch': epoch
                        }, best_checkpoint_path)
                        print(f"Saved checkpoint at {best_checkpoint_path} with AUC: {auc:.4f}")
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"Early stopping at step {step}")
                            break
                    model.train()
            if patience_counter >= patience:
                break

        if best_checkpoint_path:
            checkpoint = torch.load(best_checkpoint_path, weights_only=False)
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print(f"Loaded best model from {best_checkpoint_path} with AUC: {checkpoint['auc']:.4f}")

        auc, accuracy, precision, recall = metrics_evaluator.evaluate(model, config.original_data_path, config.num_workers, evaluate_dataset)
        print(f"Final AUC: {auc:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

class MainExecutor:
    def __init__(self):
        self.seed = 42
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def run(self):
        print('Using device:', device)
        trainer = Trainer()
        trainer.train()

if __name__ == '__main__':
    print('Using device:', device)
    executor = MainExecutor()
    executor.run()

#python pixu_train.py --data_dir pixu/preprocessed_data --algorithm core
#python pixu_train.py --data_dir pixu/preprocessed_data --algorithm visual
#python pixu_train.py --data_dir pixu/preprocessed_data --algorithm category
#python pixu_train.py --data_dir pixu/preprocessed_data --algorithm temporal
#python pixu_train.py --data_dir pixu/preprocessed_data --algorithm personalized