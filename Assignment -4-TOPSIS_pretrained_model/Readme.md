# TOPSIS-Based Selection of Best Pre-Trained Text Classification Model

## Roll Number: 102303982

## Assignment Category: Text Classification

---

# 1. Introduction

As per the assignment mapping:

- Roll numbers ending with **2 or 7 → Text Classification**

Since my roll number is **102303982**, this project applies the **TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)** method to determine the best pre-trained text classification model.

The goal is to evaluate multiple transformer-based models using:

- Performance metrics (accuracy, F1-score)
- Efficiency metrics (latency, model size, memory usage)

TOPSIS is used to select the model that provides the best balance between performance and deployability.

---

# 2. Models Evaluated

The following pre-trained HuggingFace models were evaluated on the SST-2 dataset:

1. `distilbert-base-uncased-finetuned-sst-2-english`
2. `textattack/bert-base-uncased-SST-2`
3. `siebert/sentiment-roberta-large-english`
4. `cardiffnlp/twitter-roberta-base-sentiment-latest`
5. `nlptown/bert-base-multilingual-uncased-sentiment`

---

# 3. Dataset Used

- Dataset: GLUE SST-2
- Task: Binary Sentiment Classification
- Split Used: Validation
- Samples Evaluated: 500

---

# 4. Evaluation Criteria

Each model was evaluated using the following criteria.

## 4.1 Quality Criteria (Higher is Better)

| Criterion | Description                           |
| --------- | ------------------------------------- |
| Accuracy  | Classification accuracy               |
| F1 Score  | Harmonic mean of precision and recall |

## 4.2 Efficiency Criteria (Lower is Better)

| Criterion             | Description                        |
| --------------------- | ---------------------------------- |
| Latency (ms)          | Average inference time per sample  |
| Parameters (Millions) | Model size                         |
| VRAM (GB)             | Approximate GPU memory requirement |

---

# 5. Decision Matrix

| Model                       | Accuracy | F1    | Latency (ms) | Params (M) | VRAM (GB) |
| --------------------------- | -------- | ----- | ------------ | ---------- | --------- |
| DistilBERT                  | 0.912    | 0.918 | 28.98        | 66         | 0.5       |
| BERT-base (textattack)      | 0.934    | 0.938 | 54.21        | 110        | 0.8       |
| RoBERTa-large (siebert)     | 0.930    | 0.934 | 192.69       | 355        | 1.3       |
| RoBERTa-base (cardiffnlp)   | 0.814    | 0.795 | 52.30        | 125        | 0.9       |
| BERT-multilingual (nlptown) | 0.750    | 0.794 | 39.10        | 110        | 0.8       |

_Note: Model parameters and VRAM values are taken from HuggingFace model documentation._

---

# 6. Weights and Impacts

The following weights were assigned:

| Criterion  | Weight | Impact |
| ---------- | ------ | ------ |
| Accuracy   | 0.30   | +      |
| F1 Score   | 0.25   | +      |
| Latency    | 0.20   | -      |
| Parameters | 0.15   | -      |
| VRAM       | 0.10   | -      |

- "+" indicates higher values are preferred.
- "-" indicates lower values are preferred.

Performance was prioritized over efficiency.

---

# 7. TOPSIS Methodology

TOPSIS follows these steps:

1. Construct the decision matrix.
2. Normalize the matrix using vector normalization.
3. Multiply normalized matrix by weights.
4. Determine the ideal best and ideal worst solutions.
5. Compute Euclidean distances from ideal best and ideal worst.
6. Compute the TOPSIS score:

Score = D⁻ / (D⁺ + D⁻)

Where:

- D⁺ = Distance from ideal best
- D⁻ = Distance from ideal worst

The model with the highest score is considered the optimal choice.

---

# 8. How to Run the Project

## Step 1: Install Dependencies

pip install -r requirements.txt

## Step 2: Evaluate Models

python src/evaluate_modles.py

This generates:

outputs/evaluation_results.csv

Move the file to:

data/results.csv

Fill the `params_millions` and `vram_gb` columns manually.

---

## Step 3: Run TOPSIS

PowerShell single-line command:

python src/run_topsis.py --input data/results.csv --criteria accuracy,f1,latency_ms,params_millions,vram_gb --weights 0.30,0.25,0.20,0.15,0.10 --impacts +,+,-,-,- --out_csv outputs/topsis_ranking.csv --out_plot outputs/topsis_scores.png

This generates:

- outputs/topsis_ranking.csv
- outputs/topsis_scores.png

---

# 9. Results

The ranked results are available in:

outputs/topsis_ranking.csv

Graph:

![TOPSIS Scores](outputs/topsis_scores.png)

---

# 10. Conclusion

Using TOPSIS, the model with the highest score is selected as the best overall model considering both:

- Predictive performance
- Computational efficiency
- Practical deployability

This approach ensures a balanced decision rather than choosing based solely on accuracy.

---

# 11. Key Learnings

- Model selection requires multi-criteria evaluation.
- Accuracy alone is insufficient for production systems.
- Efficiency metrics significantly affect real-world deployment.
- TOPSIS provides a structured decision-making framework.

---

# 12. Repository Structure

topsis-text-classification/
│
├── data/
│ └── results.csv
│
├── src/
│ ├── evaluate_modles.py
│ └── run_topsis.py
│
├── outputs/
│ ├── evaluation_results.csv
│ ├── topsis_ranking.csv
│ └── topsis_scores.png
│
├── requirements.txt
└── README.md
