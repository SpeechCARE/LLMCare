# Component 1 – Transformer & Fusion Screening

## Overview

This component develops a **screening algorithm** for detecting cognitive decline using both **transformer-based embeddings** and **handcrafted linguistic features**, combined through a **fusion classifier**.

* **Transformer Baselines**: Evaluated 10 pretrained models (general-purpose and biomedical/clinical). Compared frozen, last-layer, and full fine-tuning strategies.
* **Handcrafted Features**: 110 lexical, syntactic, semantic, and psycholinguistic features for interpretability.
* **Fusion Classifier**: Combines transformer embeddings with handcrafted features via a late fusion strategy for improved accuracy and robustness.

## Project Structure

```
├── Config/
│   ├── __init__.py
│   └── config.py            # Hyperparameters (batch size, learning rate, etc.)
│
├── data/
│   ├── __init__.py
│   └── dataset.py           # Dataset loading and preprocessing
│
├── Model/
│   ├── __init__.py
│   └── model.py             # Transformer, MLP, and fusion model definitions
│
├── utils/
│   ├── __init__.py
│   ├── global_constants.py  # Data paths and directory constants
│   ├── supplementary.py     # Extra utilities (e.g., metrics, feature extraction)
│   └── utils.py             # Helper functions (logging, checkpointing, etc.)
│
├── notebooks/
│   └── extract_handcrafted_features.ipynb  # Run this before main.py
│
├── main.py                  # Entry point script
├── train.py                 # Training loop
├── requirements.txt         # Dependencies
└── README.md
```

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running

### Step 1 – Extract Handcrafted Features

Before training or running the pipeline, you must generate the handcrafted linguistic features.
Open and run the notebook:

```bash
notebooks/extract_handcrafted_features.ipynb
```

This will produce a feature file (e.g., features.csv) that is later combined with transformer embeddings.

### Step 2 – Train / Run Pipeline

Run the main script:

```bash
python main.py
```

---

## 🔧 Configuration

You can customize the experiment in two places:

### 1. **Config/config.py** – Hyperparameters

Here you control model and training settings.

**Example snippet:**

```python
# Config/config.py
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 50
DROPOUT = 0.4
MODEL_NAME = "bert-base-uncased"
```

* Change `MODEL_NAME` to try different transformers (`roberta-base`, `distilbert-base-uncased`, etc.).
* Increase/decrease `BATCH_SIZE` depending on GPU memory.
* Adjust `LEARNING_RATE` or `EPOCHS` to balance training stability vs. overfitting.

---

### 2. **utils/global_constants.py** – Data Paths & Globals

This file stores dataset paths and directory constants.

**Example snippet:**

```python
# utils/global_constants.py
TRAIN_DATA_PATH = "./data/train.csv"
VAL_DATA_PATH   = "./data/val.csv"
TEST_DATA_PATH  = "./data/test.csv"

CHECKPOINT_DIR = "./checkpoints/"
LOG_DIR        = "./logs/"
```

* Update these paths if your dataset is stored elsewhere.
* Change `CHECKPOINT_DIR` to decide where trained models are saved.
* Set `LOG_DIR` for TensorBoard or training logs.
