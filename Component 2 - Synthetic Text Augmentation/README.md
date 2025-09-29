
# Component 2 - Synthetic Text Augmentation

This repository contains two complementary Jupyter notebooks for **synthetic text generation and fine-tuning large language models (LLMs)**. The project focuses on generating label-conditioned synthetic speech data to augment training datasets for cognitive impairment classification.

---

## ✨ Key Components

### 🔹 Synthetic Data Generation Framework

We adopt a **label-conditioned language modeling framework**, modeling conditional probabilities of token sequences given a cognitive label.

* **Healthy (Control):** Generates fluent speech with advanced syntax and structured storytelling.
* **Cognitively Impaired (Case):** Generates speech with repetitions, filler words, hesitations, and fragmented clauses.

### 🔹 LLMs Evaluated

We fine-tuned and benchmarked a variety of models:

* **LLaMA 3.1 8B Instruct** – balanced size and quality, good for class-specific variation.
* **MedAlpaca 7B** – clinically fine-tuned, tested for biomedical linguistic alignment.
* **Ministral 8B Instruct** – fast, fluent outputs, typical of non-impaired speech.
* **LLaMA 3.3 70B Instruct** – tested for scaling effects on complex disorganized speech.
* **GPT-4o (text-only)** – benchmark for fluency and subtle disfluencies.

### 🔹 Fine-Tuning Strategy

* **Framework:** QLoRA (Quantized Low-Rank Adapters) for efficient training.
* **Adapters:** Inserted into linear/QKV layers depending on model size.
* **Hyperparameters:**

  * LoRA ranks: 64 & 128
  * Dropout: 0 or 0.1
  * Optimizer: PagedAdamW, mixed-precision (float16)
  * Scheduler: Cosine learning rate
  * LR: 2e-4 or 1e-4 (open-weight models), tuned multipliers for GPT-4o
  * Training: 10 epochs

### 🔹 Prompt Design & Inference

* **Training Prompts:** Label-specific role-based prompts (e.g., *“language and cognition specialist”* vs. *“speech pathologist”*) with 10 variations to improve diversity.
* **Inference Prompts:** Shifted to **neutral prompts** to encourage spontaneous outputs instead of repetitive label-driven speech.
* **Hyperparameters:** Tuned for balance of coherence and diversity (`top-p`, `top-k`, `temperature`).

---

## 📂 Notebooks

### 1. **`Generation_gpt4o_finetuning.ipynb`**

A pipeline for **synthetic data generation and fine-tuning GPT-4o**.

* Prepares label-conditioned JSONL datasets from raw CSVs.
* Defines prompts for generating cognitively healthy vs. impaired speech.
* Validates dataset formatting.
* Uploads training/validation sets to OpenAI.
* Runs fine-tuning jobs, monitors progress, and retrieves tuned models.
* Supports checkpointing and data augmentation via inference.

### 2. **`TextGenerationPipeline_For_OpenSource_Models.ipynb`**

A pipeline for **fine-tuning and inference with open-source LLMs**.

* Installs and configures Hugging Face, PEFT, TRL, and bitsandbytes.
* Defines `ModelHandler` (model loading & LoRA fine-tuning) and `DatasetHandler` (CSV → Hugging Face dataset).
* Sets up a **training pipeline** with Hugging Face `Trainer`.
* Evaluates fine-tuned models.
* Runs inference and saves outputs.
* Saves and exports fine-tuned models for reuse or sharing on Hugging Face Hub.

---

## 🚀 How to Use

1. Open each notebook in **JupyterLab** or **Google Colab**.
2. Run setup cells (`pip install`, imports, logging).
3. Prepare your datasets (`train.csv`, `validation.csv`, `test.csv`).
4. Follow the pipeline: dataset preparation → fine-tuning → evaluation → inference.
5. Collect synthetic augmentations and use them in downstream classification models.

---

## 📌 Notes

* Synthetic data should be **used only for training augmentation**.
* Validation and test sets must always remain **real, held-out participants**.
* Ensure API keys (`OpenAI`, `wandb`, `Hugging Face Hub`) are correctly configured.