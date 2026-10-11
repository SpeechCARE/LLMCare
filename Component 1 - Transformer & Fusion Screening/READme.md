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

## Environment Setup (Conda)

Create a dedicated Conda environment named `speechcare` with Python 3.11
before installing the project dependencies.

These instructions assume a Linux x86-64 server with Bash, Git, wget,
and access to Jupyter Notebook or JupyterLab. The same workflow can be
used on RunPod, Vast.ai, and other servers that provide these tools.

The examples use `/workspace`. If this directory is unavailable or
not writable, choose another directory, such as `$HOME/workspace`,
and update the paths below accordingly.

**Google Colab:** To use this same Conda environment through the Colab
interface, connect to a Jupyter runtime launched from `speechcare`,
following Google's
[local-runtime instructions](https://research.google.com/colaboratory/local-runtimes.html).
Creating and registering a Conda environment does not automatically
change the interpreter of a standard hosted Colab notebook.

### Step 1 – Install Miniconda

Run the following commands in a **Bash terminal**.

Set your workspace directory:

```bash
export SPEECHCARE_WORKSPACE="/workspace"

mkdir -p "$SPEECHCARE_WORKSPACE"
cd "$SPEECHCARE_WORKSPACE"
```

Install Miniconda once. If Miniconda is already installed at the path
below, skip the download and installer commands.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh \
  -b -p "$SPEECHCARE_WORKSPACE/miniconda3"
```

Activate Miniconda and initialize Bash:

```bash
source "$SPEECHCARE_WORKSPACE/miniconda3/bin/activate"

conda init bash
source ~/.bashrc

conda --version
```

### Step 2 – Create the speechcare Environment

Continue in the same Bash terminal.

After reviewing and agreeing to the Terms of Service for the specified
Anaconda channels, accept them:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

Create and activate the environment:

```bash
conda create -n speechcare python=3.11 pip -y

conda activate speechcare
```

Install and register the notebook kernel:

```bash
python -m pip install ipykernel

python -m ipykernel install --user \
  --name speechcare \
  --display-name "Python (speechcare)"
```

If the `speechcare` environment already exists, activate it instead
of running the environment-creation command again.

### Step 3 – Select the Notebook Kernel

In Jupyter Notebook or JupyterLab, open the kernel selector and choose
**Python (speechcare)**. Refresh Jupyter if the new kernel is not
initially listed.

Run the following in a notebook code cell to verify the interpreter:

```python
import sys

print(sys.executable)
print(sys.version)
```

For the default installation path, the executable should be:

```text
/workspace/miniconda3/envs/speechcare/bin/python
```

The Python version should be 3.11.

Select this kernel for both feature extraction and pipeline execution.

## Installation

### Step 1 – Download the Repository

In the same Bash terminal, clone the repository into your workspace:

```bash
cd "$SPEECHCARE_WORKSPACE"

git clone https://github.com/SpeechCARE/LLMCare.git
```

If the repository is already present, use the existing copy.

### Step 2 – Install Project Dependencies

Open a notebook using **Python (speechcare)**.

In a notebook code cell, enter the Component 1 directory and install
the dependencies:

```python
%cd "/workspace/LLMCare/Component 1 - Transformer & Fusion Screening"

%pip install -r requirements.txt
```

If you chose a different workspace directory, replace `/workspace`
in the notebook commands with that path.

If installation requests a kernel restart, restart
**Python (speechcare)** and rerun the directory-change command.

## Running

Before running the notebooks or pipeline, update the dataset and
output paths in `utils/global_constants.py` and review the settings
in `Config/config.py`. See the Configuration section below.

### Step 1 – Generate Handcrafted Features

Open:

```text
notebooks/extract_handcrafted_features.ipynb
```

Select **Python (speechcare)**, update any input and output paths
required by the notebook, and run its cells to generate the
handcrafted features used by the fusion pipeline.

Note: If you plan to use only the Transformer-based models without fusing their representations with handcrafted linguistic features, you can skip this step and proceed directly to Step 2.

### Step 2 – Run the Pipeline

Open a notebook using **Python (speechcare)**.

Run the following in a notebook code cell, adjusting the workspace
path if necessary:

```python
%cd "/workspace/LLMCare/Component 1 - Transformer & Fusion Screening"

%run main.py
```

`%run main.py` executes the script using the notebook's selected
Python environment.

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
