# Component 4 – Multimodal LLMs (Qwen & Phi)

## Overview

This component explores fine-tuning and evaluating **state-of-the-art multimodal models** for audio–text tasks:

* **Qwen 2.5 Omni**

  * “Thinker Talker” architecture handling text, audio, image, and video.
  * Supports real-time speech responses.
  * Fine-tuning via Hugging Face checkpoints and LLaMA-Factory.

* **Phi-4-Multimodal**

  * Microsoft’s multimodal model unifying speech, vision, and language encoders.
  * Supports **128K token context**.
  * Used in open-weight form for domain-specific fine-tuning.

We evaluated both **Qwen** and **Phi-4** in **zero-shot** and **fine-tuned** settings.

---

## 📂 Project Structure

```
Component 4 – Multimodal LLMs
├── Phi4/
│   ├── finetune.py
│   ├── requirements.txt
│   ├── run.sh
│   ├── test.py
│   └── test.sh
│
└── qwen/
    ├── create_dataset_json.py
    ├── test_audio_classification_2label.py
    ├── test_audio_classification_multiclass.py
    ├── test_requirements.txt
    └── train.yaml
```

---

## Qwen

### Setup

```bash
cd qwen
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

### 📂 Dataset Preparation

Convert your dataset into JSON format:

```bash
python qwen/create_dataset_json.py
```

### Training

Train using the provided YAML config:

```bash
llamafactory-cli train qwen/train.yaml
```

### Inference

Fist install dependencies:

```bash
pip install -r ../qwen/test_requirements.txt   # test dependencies
```

Then, run inference for audio classification:

```bash
python qwen/test_audio_classification.py
```

---

## Phi-4

### Setup

```bash
cd Phi4
pip install -r requirements.txt
```

### Training

Run fine-tuning with the shell script:

```bash
bash finetune.sh
```

### Inference

Run inference with:

```bash
bash test.sh
```

or directly:

```bash
python test.py
```
