# Hinglish YouTube Sentiment Analysis

Deep learning-based sentiment classification of code-mixed Hinglish (Hindi-English) YouTube comments, comparing four RNN architectures (RNN, LSTM, GRU, BiLSTM) against a DistilBERT transformer baseline.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/transformers/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/hinglish-youtube-sentiment-analysis/blob/main/sentiment_analysis.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Visualizations](#visualizations)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architectures](#model-architectures)
- [Results & Analysis](#results--analysis)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Overview

Hinglish — romanized Hindi mixed with English — is one of the most common forms of communication on Indian social media. Standard NLP and sentiment analysis tools, which are mostly trained on monolingual English data, perform poorly on this kind of code-mixed text because of informal grammar, inconsistent transliteration, and mixed-script vocabulary.

This project builds a complete sentiment analysis pipeline for Hinglish YouTube comments, from raw data collection to model evaluation, and answers the following question:

> **Which recurrent neural network architecture offers the best balance of accuracy and computational efficiency for Hinglish sentiment classification, and how does it compare to a transformer-based model?**

**Project Highlights**

- Collected 24,000 real YouTube comments from 12 videos across entertainment, music, and vlog categories
- Built an end-to-end NLP pipeline: collection → cleaning → auto-labeling → balancing → tokenization → modeling
- Trained and benchmarked four RNN variants (Simple RNN, LSTM, GRU, BiLSTM) against a fine-tuned DistilBERT model
- Compared models on accuracy, precision, recall, F1-score, training time, and memory usage
- Translated results into practical deployment recommendations for different real-world scenarios

---

## Key Results

<img width="1472" height="537" alt="image" src="https://github.com/user-attachments/assets/498c6fcb-df6a-48b9-b00f-4fddc2b7489c" />


---

## Visualizations
<img width="937" height="232" alt="image" src="https://github.com/user-attachments/assets/5283ea02-9d98-4b8a-9937-bc35186bf781" />
<img width="635" height="382" alt="image" src="https://github.com/user-attachments/assets/fda291b1-4200-48b4-8cb9-0f0b41731593" />


The notebook generates the following charts and tables, saved under `assets/`:

| Visualization | Description |
|---|---|
| `dataset_distribution.png` | Class distribution before and after balancing |
| `sentiment_pie_charts.png` | Overall sentiment breakdown of the dataset |
| `chart_training_curves.png` | Accuracy and loss curves for all models |
| `table_complete_comparison.png` | Full metrics comparison across models |
| `table_efficiency.png` | Training time and memory usage comparison |
| `chart_dashboard.png` | Combined results dashboard |

> Add these images directly to the `assets/` folder after running the notebook so they render here on GitHub.

---

## Dataset

### Collection

- **Source:** YouTube comments, scraped using `youtube-comment-downloader`
- **Videos:** 12, spanning entertainment, music, and vlog categories
- **Comments per video:** 2,000 (sorted by popularity)
- **Total raw comments collected:** 24,000

### Preprocessing Pipeline

| Stage | Action | Rationale |
|---|---|---|
| Deduplication | Remove exact duplicate comments | Prevents data leakage between train/test |
| Null removal | Drop empty or null comments | Ensures every sample is usable |
| Lowercasing | Convert all text to lowercase | Reduces vocabulary size |
| URL removal | Strip `http`/`www` links | URLs carry no sentiment information |
| Mentions & hashtags | Remove `@user` and `#tag` | Social noise unrelated to sentiment |
| Emoji conversion | `emoji.demojize()` | Converts emojis to text so sentiment is preserved |
| Mild cleaning | Retain `!`, `?`, `.`, `,`, digits | Punctuation carries sentiment intensity |
| Readability filter | Remove non-ASCII-only comments | Keeps romanized Hinglish, drops pure Devanagari/other scripts |

**Final dataset:** 9,552 balanced comments — 3,184 each for Positive, Negative, and Neutral classes.
<img width="1405" height="537" alt="image" src="https://github.com/user-attachments/assets/3ac88682-da47-4f37-a755-e27dcb5f8411" />


### Labeling

- Automated sentiment labeling using `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Processed in batches of 32 comments with 512-token truncation
- Three-class output: Positive, Negative, Neutral
<img width="1365" height="557" alt="image" src="https://github.com/user-attachments/assets/db6a69a5-e638-4cc0-b909-d0f93ff11b45" />

---

## Methodology

### 1. Data Collection

```python
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR

downloader = YoutubeCommentDownloader()
video_urls = [
    "https://youtu.be/VIDEO_ID_1",
    "https://youtu.be/VIDEO_ID_2",
    # ... 12 videos total
]

all_comments = []
for url in video_urls:
    count = 0
    for comment in downloader.get_comments_from_url(url, sort_by=SORT_BY_POPULAR):
        all_comments.append({'comment': comment['text'], 'source_url': url})
        count += 1
        if count >= 2000:
            break
```

### 2. Text Cleaning

```python
import re
import emoji

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r'[^\w\s!?.,0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

### 3. Train-Test Split

- **Ratio:** 80% train, 20% test
- **Stratification:** Preserves class distribution across splits
- **Random state:** 42 (for reproducibility)

### 4. Tokenization & Padding

- **Tokenizer:** Keras `Tokenizer`, vocabulary size = 20,000, with OOV token
- **Sequence length:** 95th percentile of comment length + 5 tokens (~60 tokens)
- **Padding:** Post-padding with post-truncation

---

## Model Architectures

### Common Configuration

| Setting | Value |
|---|---|
| Embedding dimension | 128 |
| Hidden units | 64 |
| Output layer | `Dense(3, activation='softmax')` |
| Loss function | `sparse_categorical_crossentropy` |
| Optimizer | Adam (default parameters) |
| Early stopping | patience = 3 on validation loss, restores best weights |

### RNN (Baseline)

```
Embedding(20000, 128) -> SimpleRNN(64) -> Dense(3, softmax)
```

- Serves as the minimum performance baseline
- Known limitation: vanishing gradients on longer sequences

### LSTM

```
Embedding(20000, 128) -> LSTM(64) -> Dense(3, softmax)
```

- Memory gates (input, forget, output) mitigate vanishing gradients
- Captures long-range dependencies, e.g. negation patterns like "not bad"

### GRU

```
Embedding(20000, 128) -> GRU(64) -> Dense(3, softmax)
```

- Merged update gate reduces parameter count by ~25% compared to LSTM
- Faster training with comparable memory capacity

### BiLSTM

```
Embedding(20000, 128) -> Bidirectional(LSTM(64)) -> Dense(3, softmax)
```

- Processes sequences in both directions
- Particularly useful for sentiment, where later words can modify the meaning of earlier ones
- **Best RNN variant overall:** 74% accuracy, converges in 5 epochs

### DistilBERT (Transformer Baseline)

- Pretrained `distilbert-base-uncased` with a 3-class classification head
- Learning rate: `2e-5`, Batch size: `16`, Epochs: `3`
- 40% smaller and 60% faster than BERT while retaining ~97% of its accuracy

---

## Results & Analysis


### Training Behavior

- **RNN:** Severe overfitting (train accuracy ~90% vs. validation ~65%), confirming poor generalization
- **LSTM / GRU:** Healthy convergence with a train-validation gap of ~10–15%; GRU trains faster than LSTM
- **BiLSTM:** Best validation accuracy (72.95%) with the fastest convergence among RNNs (5 epochs)
<img width="822" height="612" alt="image" src="https://github.com/user-attachments/assets/36cdff11-f2d0-4a68-be6f-8a37085b4b05" />

### Efficiency Trade-offs

- **Fast, moderate-accuracy cluster:** RNN, GRU, BiLSTM — all under 15 seconds training time, 62–72.9% accuracy
- **Slow, high-accuracy cluster:** DistilBERT — 250 seconds training time, 85% accuracy
- <img width="1182" height="570" alt="image" src="https://github.com/user-attachments/assets/eb01b5ec-7cd4-48d3-a4b3-bb562b86113f" />


### Per-Class Analysis

- **Negative / Positive:** Strong separation due to clear sentiment-indicating words
- **Neutral:** The most frequently misclassified class, due to ambiguous intensity and language mixing
- Both BiLSTM and DistilBERT struggle with the neutral class, suggesting this is a dataset-level limitation rather than a model-specific one

### Deployment Recommendations

| Scenario | Recommended Model | Rationale |
|---|---|---|
| Real-time comment moderation | BiLSTM | Best accuracy-to-time ratio |
| Resource-constrained / edge device | GRU | Fastest inference, smallest footprint |
| Offline batch processing | DistilBERT | Highest accuracy justifies the extra cost |
| Baseline / proof-of-concept | RNN | Simplest to implement and benchmark against |

---

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Dependencies

- Python >= 3.10
- TensorFlow >= 2.12
- PyTorch >= 2.0
- Transformers >= 4.30
- pandas, numpy, matplotlib, seaborn, scikit-learn
- youtube-comment-downloader, emoji, langdetect, psutil

### GPU Support

The notebook is optimized for Google Colab with a GPU runtime. For local execution:

```bash
# CUDA-enabled TensorFlow
pip install tensorflow[and-cuda]

# Verify GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## Usage

### Option 1: Google Colab (Recommended)

1. Open `sentiment_analysis.ipynb` in Google Colab
2. Enable GPU runtime: **Runtime → Change runtime type → GPU**
3. Mount Google Drive when prompted
4. Run all cells sequentially

### Option 2: Local Execution

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/hinglish-youtube-sentiment-analysis.git
cd hinglish-youtube-sentiment-analysis

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook sentiment_analysis.ipynb
```

### Notebook Sections

1. Data Collection (YouTube)
2. Data Preprocessing & Cleaning
3. Sentiment Labeling (RoBERTa)
4. Dataset Balancing
5. Train-Test Split & Tokenization
6. Model Training (RNN, LSTM, GRU, BiLSTM)
7. Transformer Baseline (DistilBERT)
8. Evaluation & Visualization
9. Results Analysis & Discussion
10. Conclusion

---

## Repository Structure

```
hinglish-youtube-sentiment-analysis/
├── README.md                       # Project documentation
├── LICENSE                         # MIT License
├── requirements.txt                # Python dependencies
├── sentiment_analysis.ipynb        # Main notebook (full pipeline)
├── .gitignore                      # Git ignore rules (Python)
└── assets/
    ├── dataset_distribution.png    # Before/after balancing charts
    ├── sentiment_pie_charts.png    # Class distribution pie charts
    ├── chart_training_curves.png   # Accuracy/loss vs. epochs
    ├── table_complete_comparison.png  # Full metrics table
    ├── table_efficiency.png        # Time & memory comparison
    └── chart_dashboard.png         # Combined results dashboard
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | TensorFlow / Keras, PyTorch |
| Transformers | Hugging Face Transformers (DistilBERT, RoBERTa) |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| NLP Utilities | youtube-comment-downloader, emoji, langdetect |
| Evaluation | scikit-learn (`classification_report`, `confusion_matrix`) |
| Environment | Google Colab (GPU runtime) |

---

## Limitations

- **Dataset size:** 9,552 samples is modest for deep learning; 50,000+ comments would likely improve generalization
- **Label noise:** Automated RoBERTa labels on Hinglish text are imperfect; manual verification would strengthen the ground truth
- **No pretrained Hinglish embeddings:** fastText-style Hindi-English embeddings could improve RNN performance
- **Single platform:** Results are based on YouTube comments and may not generalize directly to Twitter, Instagram, or other platforms
- **English-centric transformer:** DistilBERT has no explicit Hinglish pretraining; multilingual models may perform better

---

## Future Work

- [ ] Evaluate multilingual transformers (mBERT, XLM-RoBERTa) for native Hinglish support
- [ ] Implement data augmentation via back-translation (Hindi ↔ English)
- [ ] Add attention/explainability visualizations for model interpretability
- [ ] Deploy as a real-time API for live comment moderation
- [ ] Expand the dataset to 50,000+ comments with manual labeling
- [ ] Experiment with pretrained Hinglish word embeddings

---

## Citation

If you use this project in your research or work, please cite it as:

```bibtex
@misc{hinglish_sentiment_2025,
  title  = {Sentiment Analysis of Hinglish YouTube Comments Using Deep Learning},
  author = {Mahnoor Zakir},
  year   = {2026},
  url    = {https://github.com/Mahnoor-data/hinglish-youtube-sentiment-analysis}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For questions, suggestions, or collaboration opportunities, feel free to open an issue on this repository or reach out via [LinkedIn](https://www.linkedin.com/in/mahnoor-zakir-9a6183358?utm_source=share_via&utm_content=profile&utm_medium=member_android).

---

**Keywords:** Sentiment Analysis, Hinglish, Code-Mixed NLP, RNN, LSTM, GRU, BiLSTM, DistilBERT, Deep Learning, YouTube Comments, Natural Language Processing
