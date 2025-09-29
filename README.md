# LLMCARE: Alzheimer’s Detection via Transformer Models Enhanced by LLM-Generated Synthetic Data

## Overview

This repository implements **LLMCARE**, a speech-based NLP pipeline for early detection of Alzheimer’s disease and related dementias (ADRD). The pipeline integrates:

1. **Transformer embeddings** with handcrafted linguistic features.
2. **Synthetic data augmentation** using large language models (LLMs).
3. **LLMs as classifiers** (text-only and multimodal).
4. **External generalizability evaluation** using the **DementiaBank Delaware Corpus**.

The repository is organized into **five components**, each with its own README and runnable code.

---

## Components

### **Component 1 – Transformer & Fusion Screening**

* Evaluates **ten transformer models** (general-purpose + biomedical/clinical).
* Compares frozen, last-layer, and full fine-tuning.
* Incorporates **110 handcrafted linguistic features**.
* Develops a **fusion classifier** combining embeddings + features.
  👉 See [Component 1 README](./Component%201%20-%20Transformer%20%26%20Fusion%20Screening/README.md)

---

### **Component 2 – Synthetic Text Augmentation**

* Uses **LLMs** (e.g., LLaMA-8B/70B, MedAlpaca-7B, Ministral-8B, GPT-4o) to generate **label-conditioned synthetic speech transcripts**.
* Augmented data boosts F1 scores (e.g., MedAlpaca +2× augmentation improved F1 to 85.7).
  👉 See [Component 2 README](./Component%202%20-%20Synthetic%20Text%20Augmentation/README.md)

---

### **Component 3 – LLMs as Text-Only Classifiers**

* Evaluates **fine-tuned LLMs** as direct classifiers for speech transcripts.
* Demonstrates significant performance gains with fine-tuning.
  👉 See [Component 3 README](./Component%203%20-%20LLMs%20as%20Text-Only%20Classifiers/README.md)

---

### **Component 4 – Multimodal LLMs**

* Benchmarks **multimodal LLMs** (GPT-4o, Qwen 2.5 Omni, Phi-4) for audio-text classification.
* Includes **two subfolders**:

  * **Qwen** → Fine-tuning with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

    * Train with: `llamafactory-cli train train.yaml`
    * Inference with: `test_audio_classification_2label.py`
  * **Phi4** → Fine-tuning with native scripts.

    * Train with: `bash run.sh` or `python finetune.py`
    * Inference with: `bash test.sh` or `python test.py`
      👉 See [Component 4 README](./Component%204%20-%20Multimodal%20LLMs/README.md)

---

### **Component 5 – External Generalizability Evaluation (DementiaBank Delaware Corpus)**

* Tests pipeline generalizability on an **MCI-only cohort** (n=205).
* Tasks: Cookie Theft, Cinderella recall, procedural discourse.
* Pipeline achieved **F1 = 72.8 (AUC = 69.6)** with MedAlpaca augmentation.
* **Note:** This codebase is the **same as Component 1 (ADReSSo pipeline)**. To run, simply **update data paths** in configs to point to the Delaware dataset.

---