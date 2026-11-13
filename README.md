# PIXU — Personalized News Article Image Selection

**Research code for modeling readers' visual preferences in personalized news.**

PIXU explores a question that standard news recommenders often overlook: *when an article can be shown with different images, which image best matches a reader's interests?* It uses visual representations from CLIP and information from reading history to study personalized image selection.

**Keywords:** Recommender Systems · Computer Vision · CLIP · PyTorch · Human-Centered Evaluation

## Project at a glance

- **Goal:** Personalize an article's image selection without changing which articles the news recommender chooses.
- **Modeling:** CLIP-based visual features and a PyTorch model trained from user interaction data.
- **Evaluation:** Offline recommendation metrics, followed by an exploratory deployment in a live personalized-news newsletter.
- **Research context:** [POPROX](https://poprox.ai/) at the University of Minnesota's [GroupLens Research](https://grouplens.org/).

## Research results

The associated PIXU research reported:

| Evaluation | Result |
| --- | --- |
| Offline evaluation on MIND-Large | **68.1% accuracy**, **0.783 AUC** |
| Popularity baseline | 35.7% accuracy |
| Best classical baseline | 63.0% accuracy |
| Live field deployment | **5 weeks**, **163 participants**, **5,567 newsletter deliveries** |

The live deployment did **not** show a statistically significant change in clicks. A small follow-up experience survey (n=11) showed exploratory positive trends, but these differences were also **not statistically significant**. These distinctions matter: offline performance does not automatically imply improved user experience.

These figures describe the associated research study, **not a promise that the scripts in this repository reproduce every published experiment out of the box**.

## Repository contents

| File | Purpose |
| --- | --- |
| [`pixu_preprocess.py`](pixu_preprocess.py) | Preprocess news/behavior data; extract CLIP-based image representations; construct training and evaluation examples using several sampling approaches |
| [`pixu_train.py`](pixu_train.py) | Configure and train the PIXU model; evaluate AUC, accuracy, precision and recall; save model checkpoints |

The preprocessing script supports five sampling configurations: `core`, `visual`, `category`, `temporal`, and `personalized`.

## Exploring the code

This is a **research-code repository**, rather than a packaged, one-command demo. The scripts expect separately prepared news datasets and associated image files. Those datasets, images, and trained checkpoints are not included here.

Requirements visible in the scripts include Python, PyTorch, torchvision, Hugging Face Transformers/CLIP, pandas, NumPy, Pillow, scikit-learn, tqdm, and TensorBoard.

To inspect the available script options:

```bash
python pixu_preprocess.py --help
python pixu_train.py --help
```

The defaults refer to local folders such as `MINDsmall_train/`, `MINDsmall_dev/`, `images/`, and `pixu/preprocessed_data/`. Before attempting training, check the code's file paths and adapt them to your dataset layout. The repository does not currently provide a pinned environment or a fully validated reproduction recipe.

## Related publication

**Mahamudul Hasan, Daniel Kluver, Bart P. Knijnenburg, and Joseph A. Konstan.**  
*PIXU: Personalized News Article Image Selection.* ACM RecSys 2026 BEYOND Workshop.

## Author

**[Mahamudul Hasan](https://github.com/Mahamudul42)**  
PhD Candidate, Computer Science & Engineering · University of Minnesota Twin Cities

[Personal website](https://mahamudul.pp.ua/) · [Email](mailto:hasan181@umn.edu)

---

*This project explores how image personalization can be evaluated responsibly—both with offline metrics and with evidence from real readers.*
