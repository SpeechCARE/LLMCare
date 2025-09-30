# Component 3 - LLMs as Classifiers (Text-Only)

This folder contains three complementary Jupyter notebooks for **binary classification of spontaneous speech transcripts** using large language models (LLMs). The task is to label each transcript as **“Cognitively healthy”** or **“Cognitively impaired (ADRD)”**, evaluated in both **zero-shot** and **fine-tuned** settinrk.

---

## Overview

### Task & Evaluation Protocol

* **Task:** Binary classification of *spontaneous speech transcripts*.  
* **Control**Healthyhy (Control)* vs. *Cognitively Impaired (ADRD)*.  
* **Dataset Split:** Train / validation for model selection; **held-out official test set** for final reporting.  
* **Metrics:** Primary **F1-score** (with **95% confidence intervals**), plus **precision** and **recall**.  

### Models Evaluated (Text-Only)

* **LLaMA 3.1 8B Instruct**  
* **MedAlpaca 7B**  
* **Ministral 8B (2410)**  
* **LLaMA 3.3 70B Instruct**  
* **GPT-4o (2024-08-06, text-only)**  

### 🔹 Zero-Shot Prompting

We identified a **single effective prompt** that:  
* Assigns the model the role of a **cognitive-and-language expert**.  
* Explicitly states the input is a **transcript of spontaneous speech**.  
* Requests a **binary decision** (*“cognitively healthy”* vs *“cognitively impaired”*).  
* **Omits explicit linguistic cues**, encouraging internal reasoning rather than feature matching.  

**Inference hyperparameters:**  
* **Open-weight models:** `temperature = 0.0` (deterministic).  
* **GPT-4o:** `temperature = 0.7` (per platform guidance).  

### 🔹 Fine-Tuning Strategy

* **Prompt Consistency:** The **same prompt** is used during training and inference to promote **stability** and **consistMaximu of 13cy**.  
* **Training Duration:** **10 epochs** per model.  
* **Model Selection:** Best checkpoint chosen by **highest validation F1-score**.  
* **Search Strategy:** **Hyperparameter search mirrors Component 2** (see that component for details).  

---

## Notebooks

### 1. **`component3_zeroShotClassification.ipynb`**

A pipeline for **zero-shot evaluation** across both **open-weight** and mAPI-based** LLMs.  

* Insta dencies.  
* Manages API keys for **OpenAI**, **Google Gemini**, and **Hugging Face**.  
* Defines helper utilities for prediction post-processing.  
* Runs inference with **open-weight models (Hugging Face)**:  
  - LLaMA 3.2 3B  
  - Me 3.3 70B  
* Supports inference with **Gemini** (Google Generative AI).  
* Supports inference with **GPT-4o** (`gpt-4o-2024-08-06`) via OpenAI API.  
* Supports inference with **open-weight models via OpenRouter** (LLaMA 3.1 8B, LLaMA 3.3 70B).  
* Collects outputs, computes classification metrics (**F1, precision, recall with 95% CI**), and saves predictions.  

---

### 2. **`component3_finetuning_GPT.ipynb`**

A pipeline for **fine-tuning GPT-4o** on the ADReSSo classification task.  

* Loads training and validation CSVs, applies **system instruction prompt**, and converts to JSONL.  
* Includes dataset checks and preprocessing utilities.  
* Uploads datasets to **OpenAI fine-tuning API**.  
* Creates fine-tuning jobs (10 epochs) with validation monitoring.  
* Retrieves and visualizes **training/validation metrics**.  
* Manages **checkpoints** and selects best-performing model.  
* Runs inference with the tuned GPT-4o model, applies label-mapping utilities, and evaluaModelses performance (F1, precision, recall).  

---

### 3. **`component3_finetuning_openweights.ipynb`**

A pipeline for **fine-tuning open-weight LLMs** with hyperparameter search.  

* Installs and configures **Hugging Face Transformers, PEFT/TRL, and torch**.  
* Downloads target models (e.g., **MedAlpaca-7B**) from Hugging Face Hub.  
* Defines:  
  - `ModelHandler` for model setup and LoRA integration.  
  - `PromptConstructor` for task-specific prompt formatting.  
  - `DatasetHandler` for loading CSVs and converting to Hugging Face datasets.  
* Implements **TrainerHandler** (wrapper around Hugging Face `Trainer`) with callbacks.  
* Runs **token-level supervised fine-tuning** across multiple hyperparameter settings (learning rate, dropout, etc.), mirroring Component 2’s search strategy.  
* Selects the best checkpoint by **highest validation F1-score**.  
* Evaluates on the held-out test set and exports trained models (optional push to Hugging Face Hub).  

---

## How to Use

1. Open the notebooks in **JupyterLab** or **Google Colab**.  
2. Run the setup cells (dependencies, imports, logging).  
3. Prepare datasets (`train.csv`, `validation.csv`, `test.csv`) with transcript text and binary labels.  
4. **Zero-Shot:** Choose a model, set the specified `temperature`, run evaluation, and export metrics.  
5. **Fine-Tuning:**  
   * For GPT-4o → convert to JSONL, upload to OpenAI, ru tuning/evaluation.  
   * For open-weight models → run the Hugging beyond what’s specified here, **mirror Component 2’s hyperparameter search strategy** to maintain consistency across components.  
