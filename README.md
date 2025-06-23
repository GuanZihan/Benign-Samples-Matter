# ✨ Benign Samples Matter

Welcome to the official repository for our ICML 2025 paper:  
**_Benign Samples Matter! Fine-tuning on Outlier Benign Samples Severely Breaks Safety_**

This repo provides all code and scripts to reproduce the results from our paper, including sample selection, model fine-tuning, and safety evaluation.

---

## 🔧 Environment Setup

To get started, install the required packages:

```bash
pip install -r requirements.txt
```

Then to run the experiments with llama2

```bash
cd llama2
```

Download a LLaMA model (e.g., [`TheBloke/Llama-2-7B-Chat-fp16`](https://huggingface.co/TheBloke/Llama-2-7B-Chat-fp16)) and place it under the `ckpts/` directory:

```
ckpts/
└── Llama-2-7B-Chat-fp16/
```

---

## 🧪 Experiments Overview

Our attack pipeline consists of three main stages:

1. **Self-Inf-N Score Calculation**  
2. **Benign Sample Selection**  
3. **Fine-tuning the LLM**

---

## 🎯 Quick Demo

To run a demo using our method on the Dolly dataset, execute:

```bash
bash experiments/1.1_harmful_scores/Dolly/ours_evaluation.sh
```

This script fine-tunes LLaMA-2-7B-Chat using the top 100 high Self-Inf-N benign samples from Dolly.

> ⚠️ Make sure to add your OpenAI API key in `safety_evaluation/gpt4_eval.py` for GPT-4-based safety evaluations.

---

## 🔍 Sample Selection with Self-Inf-N

To compute Self-Inf-N scores and select top benign samples:

```bash
bash experiments/0_prepare_dataset/Dolly/prepare_ours.sh
```

This will generate a filtered subset of the dataset for fine-tuning.

---

## 📊 Baseline Comparisons

We also provide scripts for the baselines used in our paper:

### 🌀 Random Selection

```bash
bash experiments/1.1_harmful_scores/Dolly/random_selection_evaluation.sh
```

### 📚 COLM 2024

```bash
bash experiments/1.1_harmful_scores/Dolly/colm2024_evaluation.sh
```

---

## 📈 Additional Experiments

### 📏 Short Sample Analysis (Figure 3)

To evaluate how input length affects model safety:

```bash
bash experiments/1.1_harmful_scores/Dolly/random_fixed_evaluation.sh
```

You can vary the `fixed_length` parameter (1–15) to match the settings in Figure 3.

### 🔁 Continual Fine-tuning on Other Tasks

Try benign sample fine-tuning in continual learning setups:

```bash
bash experiments/1.2_other_experiments/Dolly/continuous_learning_asclepius.sh
```

---

## 🔗 Related Work

Our codebase builds on top of this excellent repository:  
👉 [Bidirectional Anchor (Princeton NLP)](https://github.com/princeton-nlp/benign-data-breaks-safety)

---

Feel free to open an issue or PR if you find something interesting—or broken. Happy experimenting!
