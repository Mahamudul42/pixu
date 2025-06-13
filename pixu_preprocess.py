import argparse
import os
import os.path as path
import random
import warnings
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
import csv

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Global device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Default paths
train_data_path = './MINDsmall_train'
dev_data_path = './MINDsmall_dev'
image_data_path = './images'

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

class DataPreprocessor:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Preprocess MIND dataset for multiple algorithms")
        parser.add_argument('--train_data_path', type=str, default=train_data_path, help='Path to training data')
        parser.add_argument('--dev_data_path', type=str, default=dev_data_path, help='Path to dev data')
        parser.add_argument('--image_data_path', type=str, default=image_data_path, help='Path to image directory')
        parser.add_argument('--pretrained_model_name', type=str, default='openai/clip-vit-base-patch32', help='CLIP model name')
        parser.add_argument('--output_dir', type=str, default='./pixu/preprocessed_data', help='Output directory for TSV files')
        parser.add_argument('--time_window', type=int, default=7, help='Time window in days for temporal algorithm')
        self.args = parser.parse_args()

        # Initialize CLIP for embeddings
        self.clip_model = CLIPModel.from_pretrained(self.args.pretrained_model_name).vision_model.to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(self.args.pretrained_model_name, use_fast=True)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26130258, 0.26130258, 0.27577711])
        ])

    def _load_image(self, img_path):
        """Load and preprocess image for CLIP."""
        try:
            img = Image.open(img_path).convert('RGB')
            return self.transform(img).to(device)
        except Exception as e:
            print(f"Warning: Failed to load image {img_path}: {str(e)}")
            return torch.zeros(3, 224, 224).to(device)

    def _get_image_embedding(self, img_tensor):
        """Compute CLIP embedding for an image tensor."""
        with torch.no_grad():
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
            features = self.clip_model(pixel_values=img_tensor).last_hidden_state[:, 0, :]
        return features.squeeze(0).cpu().numpy()

    def _compute_embeddings(self, news_df):
        """Compute CLIP embeddings for all news images."""
        embeddings = {}
        for news_id in tqdm(news_df.index, desc="Computing image embeddings"):
            img_path = path.join(self.args.image_data_path, f"{news_id}.jpg")
            if os.path.exists(img_path):
                img_tensor = self._load_image(img_path)
                embeddings[news_id] = self._get_image_embedding(img_tensor)
            else:
                embeddings[news_id] = np.zeros(768)  # Default embedding for missing images
        return embeddings

    def _compute_user_low_click_categories(self, behaviors_df, news_df):
        """Compute low-click categories per user based on click history."""
        user_clicks = defaultdict(lambda: defaultdict(int))
        for row in behaviors_df.itertuples():
            user = row.user
            clicked_news = [x.split('-')[0] for x in row.impressions if x.endswith('1')]
            for news_id in clicked_news:
                if news_id in news_df.index:
                    category = news_df.loc[news_id, 'category']
                    user_clicks[user][category] += 1

        user_low_click_cats = {}
        for user, cat_counts in user_clicks.items():
            total_clicks = sum(cat_counts.values())
            if total_clicks == 0:
                user_low_click_cats[user] = []
                continue
            # Categories with less than 10% of average clicks are considered low-click
            avg_clicks = total_clicks / len(cat_counts)
            low_click_cats = [cat for cat, count in cat_counts.items() if count < 0.1 * avg_clicks]
            user_low_click_cats[user] = low_click_cats if low_click_cats else list(cat_counts.keys())[-2:]  # Fallback
        return user_low_click_cats

    def _parse_timestamp(self, time_str):
        """Parse timestamp string to datetime object."""
        try:
            return datetime.strptime(time_str, '%m/%d/%Y %I:%M:%S %p')
        except ValueError:
            return datetime.min  # Fallback for invalid timestamps

    def generate_behavior_tsv(self, source, target, news_df, algorithm='core', embeddings=None, user_low_click_cats=None):
        """Generate behavior TSV using specified algorithm."""
        behaviors = pd.read_table(source, header=None, names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])
        behaviors['clicked_news'] = behaviors['clicked_news'].fillna(' ')
        behaviors['impressions'] = behaviors['impressions'].str.split()

        processed_rows = []
        invalid_rows = 0
        impression_id = 1
        all_news_ids = news_df.index.tolist()

        for row in tqdm(behaviors.itertuples(), total=behaviors.shape[0], desc=f"Generating behavior TSV ({algorithm})"):
            impressions = row.impressions
            if not impressions or len(impressions) < 1:
                invalid_rows += 1
                print(f"Warning: Empty or insufficient impressions for impression_id {row.impression_id}")
                continue

            clicked_news_ids = [x.split('-')[0] for x in impressions if x.endswith('1')]
            if not clicked_news_ids:
                invalid_rows += 1
                print(f"Warning: No clicked articles for impression_id {row.impression_id}")
                continue

            for pos_news_id in clicked_news_ids:
                if pos_news_id not in news_df.index:
                    invalid_rows += 1
                    print(f"Warning: Positive news_id {pos_news_id} not in news_df")
                    continue

                category = news_df.loc[pos_news_id, 'category']
                negative = []

                if algorithm == 'core':
                    # Core Image Triplet Formation: Same-category negatives
                    same_category_news = [n for n in news_df[news_df['category'] == category].index.tolist() if n != pos_news_id]
                    random.shuffle(same_category_news)
                    negative = same_category_news[:2]
                    if len(negative) < 2:
                        available_negatives = [n for n in all_news_ids if n != pos_news_id and n not in negative]
                        random.shuffle(available_negatives)
                        needed = 2 - len(negative)
                        negative.extend(available_negatives[:needed])

                elif algorithm == 'visual':
                    # Visually Aligned Negative Selection: Similar embeddings
                    same_category_news = [n for n in news_df[news_df['category'] == category].index.tolist() if n != pos_news_id]
                    if not same_category_news or pos_news_id not in embeddings:
                        negative = random.sample([n for n in all_news_ids if n != pos_news_id], 2)
                    else:
                        pos_emb = embeddings[pos_news_id]
                        similarities = [(n, np.dot(pos_emb, embeddings[n]) / (np.linalg.norm(pos_emb) * np.linalg.norm(embeddings[n]) or 1)) 
                                        for n in same_category_news if n in embeddings]
                        similarities.sort(key=lambda x: x[1], reverse=True)
                        negative = [n for n, _ in similarities[:2]]
                        if len(negative) < 2:
                            available_negatives = [n for n in all_news_ids if n != pos_news_id and n not in negative]
                            random.shuffle(available_negatives)
                            needed = 2 - len(negative)
                            negative.extend(available_negatives[:needed])

                elif algorithm == 'category':
                    # Category-Balanced Triplet Expansion: One same, one different category
                    same_category_news = [n for n in news_df[news_df['category'] == category].index.tolist() if n != pos_news_id]
                    diff_category_news = [n for n in news_df[news_df['category'] != category].index.tolist()]
                    random.shuffle(same_category_news)
                    random.shuffle(diff_category_news)
                    negative = same_category_news[:1] + diff_category_news[:1]
                    if len(negative) < 2:
                        available_negatives = [n for n in all_news_ids if n != pos_news_id and n not in negative]
                        random.shuffle(available_negatives)
                        needed = 2 - len(negative)
                        negative.extend(available_negatives[:needed])

                elif algorithm == 'temporal':
                    # Temporally Informed Image Sampling: Recent negatives
                    impression_time = self._parse_timestamp(row.time)
                    time_threshold = impression_time - pd.Timedelta(days=self.args.time_window)
                    same_category_news = [n for n in news_df[news_df['category'] == category].index.tolist() if n != pos_news_id]
                    recent_news = []
                    for n in same_category_news:
                        news_time = self._parse_timestamp(news_df.loc[n, 'time'] if 'time' in news_df.columns else row.time)
                        if news_time >= time_threshold:
                            recent_news.append((n, news_time))
                    recent_news.sort(key=lambda x: x[1], reverse=True)
                    negative = [n for n, _ in recent_news[:2]]
                    if len(negative) < 2:
                        available_negatives = [n for n in all_news_ids if n != pos_news_id and n not in negative]
                        random.shuffle(available_negatives)
                        needed = 2 - len(negative)
                        negative.extend(available_negatives[:needed])

                elif algorithm == 'personalized':
                    # Personalized Negative Contextualization: One same, one low-click category
                    same_category_news = [n for n in news_df[news_df['category'] == category].index.tolist() if n != pos_news_id]
                    low_click_cats = user_low_click_cats.get(row.user, [])
                    low_click_news = [n for n in news_df[news_df['category'].isin(low_click_cats)].index.tolist() if n != pos_news_id]
                    random.shuffle(same_category_news)
                    random.shuffle(low_click_news)
                    negative = same_category_news[:1] + low_click_news[:1]
                    if len(negative) < 2:
                        available_negatives = [n for n in all_news_ids if n != pos_news_id and n not in negative]
                        random.shuffle(available_negatives)
                        needed = 2 - len(negative)
                        negative.extend(available_negatives[:needed])

                if len(negative) != 2:
                    invalid_rows += 1
                    print(f"Warning: Failed to obtain 2 negative samples for news_id {pos_news_id}")
                    continue

                candidate_news = [pos_news_id] + negative
                positive_position = random.randint(0, 2)
                impressions_list = ['-0', '-0', '-0']
                impressions_list[positive_position] = '-1'
                impressions_news = [''] * 3
                impressions_news[positive_position] = pos_news_id
                negative_idx = 0
                for i in range(3):
                    if i != positive_position:
                        impressions_news[i] = negative[negative_idx]
                        negative_idx += 1
                impressions_str = ' '.join([f"{nid}{label}" for nid, label in zip(impressions_news, impressions_list)])

                processed_rows.append({
                    'impression_id': impression_id,
                    'user': row.user,
                    'time': row.time,
                    'clicked_news': row.clicked_news,
                    'impressions': impressions_str
                })
                impression_id += 1

        print(f"Total invalid rows skipped: {invalid_rows}")
        if not processed_rows:
            print(f"Error: No valid rows generated for {target}")
            return

        behaviors_new = pd.DataFrame(processed_rows)
        behaviors_new.to_csv(target, sep='\t', index=False, header=False)

    def parse_behaviors(self, source, target, user2int_path, news_df):
        """Parse behavior TSV to create behaviors_parsed.tsv."""
        behaviors = pd.read_table(source, header=None, names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])
        behaviors['clicked_news'] = behaviors['clicked_news'].fillna(' ')
        behaviors['impressions'] = behaviors['impressions'].apply(
            lambda x: x.strip().split() if isinstance(x, str) else []
        )

        user2int = {}
        for i, row in tqdm(behaviors.iterrows(), total=behaviors.shape[0], desc="Mapping users"):
            if row.user not in user2int:
                user2int[row.user] = len(user2int) + 1
        pd.DataFrame(user2int.items(), columns=['user', 'int']).to_csv(user2int_path, sep='\t', index=False)

        behaviors['user'] = behaviors['user'].map(user2int)
        processed_rows = []
        invalid_rows = 0
        all_news_ids = news_df.index.tolist()

        for row in tqdm(behaviors.itertuples(), total=behaviors.shape[0], desc="Processing impressions"):
            impressions = row.impressions
            if not impressions or len(impressions) != 3:
                invalid_rows += 1
                print(f"Warning: Invalid impressions length for impression_id {row.impression_id}")
                continue

            try:
                candidate_images = [x.split('-')[0] for x in impressions]
                labels = [x.split('-')[1] for x in impressions]
                if len(candidate_images) != 3 or len(labels) != 3 or labels.count('1') != 1:
                    invalid_rows += 1
                    print(f"Warning: Invalid impressions format for impression_id {row.impression_id}")
                    continue
            except IndexError:
                invalid_rows += 1
                print(f"Warning: Malformed impressions for impression_id {row.impression_id}")
                continue

            pos_news_id = next((x.split('-')[0] for x in impressions if x.endswith('1')), None)
            if not pos_news_id or pos_news_id not in news_df.index:
                invalid_rows += 1
                print(f"Warning: Invalid or missing positive news_id for impression_id {row.impression_id}")
                continue

            if not all(nid in news_df.index for nid in candidate_images):
                invalid_rows += 1
                print(f"Warning: Some candidate_images news IDs not in news_df for impression_id {row.impression_id}")
                continue

            clicked = ['0'] * 3
            for i, item in enumerate(impressions):
                if item.endswith('1'):
                    clicked[i] = '1'
                    break

            candidate_images_str = ' '.join([str(nid) for nid in candidate_images])
            if not candidate_images_str or len(candidate_images_str.split()) != 3:
                invalid_rows += 1
                print(f"Warning: Invalid candidate_images_str for impression_id {row.impression_id}")
                continue

            processed_rows.append({
                'user': row.user,
                'news_id': pos_news_id,
                'clicked_news': row.clicked_news,
                'candidate_images': candidate_images_str,
                'clicked': ' '.join(clicked)
            })

        print(f"Total invalid rows skipped: {invalid_rows}")
        if not processed_rows:
            print(f"Error: No valid rows processed for {target}")
            return

        behaviors_processed = pd.DataFrame(processed_rows)
        behaviors_processed['candidate_images'] = behaviors_processed['candidate_images'].astype(str).str.strip('()').str.replace("'", "").str.replace(",", "")
        invalid_rows_df = behaviors_processed[behaviors_processed['candidate_images'].str.contains(r'[\(\)\[\],]', na=False)]
        if not invalid_rows_df.empty:
            print(f"Warning: Found {len(invalid_rows_df)} rows with tuple-like candidate_images in behaviors_processed")
        behaviors_processed.to_csv(target, sep='\t', index=False, quoting=csv.QUOTE_MINIMAL, escapechar='\\')
        validate_tsv_file(target, 'candidate_images', expected_count=3)

    def parse_news(self, source, image_dir, target):
        """Parse news TSV to include image paths and categories."""
        news = pd.read_table(source, header=None, usecols=[0, 1], names=['id', 'category'], dtype={'id': str})
        news['image'] = news['id'].apply(lambda x: f"{x}.jpg")
        news['image_exists'] = news['image'].apply(lambda x: os.path.exists(os.path.join(image_dir, x)))
        news = news[['id', 'image', 'category']].set_index('id')
        news.to_csv(target, sep='\t')
        return news

    def preprocess(self):
        """Preprocess data for all algorithms."""
        os.makedirs(self.args.output_dir, exist_ok=True)
        train_dir = self.args.train_data_path
        dev_dir = self.args.dev_data_path
        image_dir = self.args.image_data_path

        # Parse news files
        train_news_df = self.parse_news(path.join(train_dir, 'news.tsv'), image_dir, path.join(self.args.output_dir, 'train_news_parsed.tsv'))
        dev_news_df = self.parse_news(path.join(dev_dir, 'news.tsv'), image_dir, path.join(self.args.output_dir, 'dev_news_parsed.tsv'))

        # Compute embeddings for visual algorithm
        print("Computing embeddings for train and dev news...")
        train_embeddings = self._compute_embeddings(train_news_df)
        dev_embeddings = self._compute_embeddings(dev_news_df)

        # Compute user low-click categories for personalized algorithm
        train_behaviors = pd.read_table(path.join(train_dir, 'behaviors.tsv'), header=None, names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])
        dev_behaviors = pd.read_table(path.join(dev_dir, 'behaviors.tsv'), header=None, names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])
        train_user_low_click_cats = self._compute_user_low_click_categories(train_behaviors, train_news_df)
        dev_user_low_click_cats = self._compute_user_low_click_categories(dev_behaviors, dev_news_df)

        # Define algorithms and their output filenames
        algorithms = [
            ('core', 'Core Image Triplet Formation'),
            ('visual', 'Visually Aligned Negative Selection'),
            ('category', 'Category-Balanced Triplet Expansion'),
            ('temporal', 'Temporally Informed Image Sampling'),
            ('personalized', 'Personalized Negative Contextualization')
        ]

        # Process train and dev splits for each algorithm
        for algo_key, algo_name in algorithms:
            print(f"\nProcessing {algo_name}...")
            # Train split
            train_behavior_tsv = path.join(self.args.output_dir, f'behavior_train_{algo_key}.tsv')
            train_behaviors_parsed_tsv = path.join(self.args.output_dir, f'behaviors_parsed_train_{algo_key}.tsv')
            train_user2int_tsv = path.join(self.args.output_dir, f'user2int_train_{algo_key}.tsv')
            self.generate_behavior_tsv(
                path.join(train_dir, 'behaviors.tsv'),
                train_behavior_tsv,
                train_news_df,
                algorithm=algo_key,
                embeddings=train_embeddings,
                user_low_click_cats=train_user_low_click_cats
            )
            self.parse_behaviors(
                train_behavior_tsv,
                train_behaviors_parsed_tsv,
                train_user2int_tsv,
                train_news_df
            )

            # Dev split
            dev_behavior_tsv = path.join(self.args.output_dir, f'behavior_dev_{algo_key}.tsv')
            dev_behaviors_parsed_tsv = path.join(self.args.output_dir, f'behaviors_parsed_dev_{algo_key}.tsv')
            dev_user2int_tsv = path.join(self.args.output_dir, f'user2int_dev_{algo_key}.tsv')
            self.generate_behavior_tsv(
                path.join(dev_dir, 'behaviors.tsv'),
                dev_behavior_tsv,
                dev_news_df,
                algorithm=algo_key,
                embeddings=dev_embeddings,
                user_low_click_cats=dev_user_low_click_cats
            )
            self.parse_behaviors(
                dev_behavior_tsv,
                dev_behaviors_parsed_tsv,
                dev_user2int_tsv,
                dev_news_df
            )

if __name__ == '__main__':
    print('Using device:', device)
    preprocessor = DataPreprocessor()
    preprocessor.preprocess()