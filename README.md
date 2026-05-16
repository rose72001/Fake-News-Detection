# Multi-Class News Classification & Explainable AI Pipeline

A research-oriented NLP framework that evaluates and compares traditional machine learning with transformer-based architectures for multi-class news topic classification. Beyond standard training, this project implements advanced research components including **Explainable AI (XAI)**, **Attention Visualization**, **Adversarial Robustness Testing**, **Bias Analysis**, and **Model Compression**.

---

## 📌 Project Objective
The goal of this project is to build a highly accurate news classification system while ensuring model transparency, robustness, and computational efficiency. The pipeline is evaluated on the **AG News Dataset**, classifying articles into four distinct categories:
* `0`: World
* `1`: Sports
* `2`: Business
* `3`: Sci/Tech

---

## 🤖 Models & Performance Comparison

The framework benchmarks three distinct architectural approaches, demonstrating a significant leap in contextual understanding moving from traditional ML to deep learning.

| Model Architecture | Accuracy | F1 Score (Weighted) | Training Complexity | Inference Speed |
| :--- | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression** (Baseline) | 91.67% | 91.65% | Low | Fast |
| **RoBERTa-Base** (Fine-Tuned) | 93.15% | 93.15% | High | Medium |
| **DistilBERT** (Compressed) | 93.43% | 93.42% | Medium | Fast |

---

## 🔬 Advanced Research Components

### 1. Explainable AI (via LIME)
To prevent the deep learning models from acting as "black boxes," a **LimeTextExplainer** is utilized. It perturbs token inputs to calculate feature importance weights, visually highlighting exactly which words (e.g., specific political, sporting, or corporate entities) drive the model's categorical predictions.

### 2. Attention Visualization (via BertViz)
Using `bertviz.head_view`, the project extracts and displays the multi-head self-attention matrices from RoBERTa’s encoder layers. This visualizes exactly how the transformer routes contextual relationships between words across different heads during inference.

### 3. Adversarial Robustness Testing
Evaluated model stability against synthetic textual perturbations (e.g., human-like spelling mistakes and typos). 
* *Example:* Testing behavior shifts between `"Apple releases new AI-powered technology..."` and the adversarial text `"Aple releeses new AI powred technologee..."`.

### 4. Bias Analysis
Tested the model's neutrality and behavioral consistency by feeding it politically and geographically varied prompts (e.g., shifting contexts between US, Chinese, European, and Indian geopolitical statements) to check for underlying category prediction shifts.

### 5. Model Compression (Knowledge Distillation)
To optimize the pipeline for real-world deployment, the project explores model compression using **DistilBERT-base-uncased**. DistilBERT retains over 99% of the baseline transformer performance while significantly increasing inference speeds and reducing memory footprints.

---

## 🛠️ Tech Stack & Libraries
* **Core DL:** `PyTorch`, `Hugging Face Transformers` (`Trainer`, `SequenceClassification`)
* **Dataset Handling:** `Hugging Face Datasets`, `Pandas`, `NumPy`
* **Traditional ML:** `Scikit-Learn` (`TfidfVectorizer`, `LogisticRegression`)
* **Interpretability & Visualization:** `LIME`, `BertViz`, `Matplotlib`

---

## 🚀 How to Run the Project

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   cd your-repo-name
