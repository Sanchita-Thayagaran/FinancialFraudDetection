# Financial Fraud Detection Pipeline

An end-to-end machine learning pipeline for detecting fraudulent financial transactions using the ULB Credit Card Fraud Detection dataset.

## Results

| Model | AUC | CV AUC | Average Precision |
|-------|-----|--------|-------------------|
| XGBoost | **96.2%** | 95.8% ± 0.3% | **0.84** |
| Random Forest | 94.7% | 94.1% ± 0.4% | 0.79 |

## Dataset

**ULB Credit Card Fraud Detection** — real anonymised transactions from European cardholders (September 2013).

| Property | Value |
|----------|-------|
| Rows | 284,807 transactions |
| Fraud cases | 492 (0.172%) |
| Features | V1–V28 (PCA-transformed), Time, Amount |
| Target | `Class` — 1 = fraud, 0 = legitimate |
| Source | [kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |

Features V1–V28 are PCA-transformed by the dataset authors to protect cardholder privacy. `Time` (seconds since first transaction) and `Amount` are the only raw features available for engineering.

## Problem

Financial fraud detection is a classic **severe class imbalance** problem:
- Only 0.172% of transactions are fraudulent
- A naive model that always predicts "not fraud" achieves 99.8% accuracy but catches zero fraud
- **False negatives** (missed fraud) cost far more than false positives (false alarms)

This pipeline solves those challenges with:
- SMOTE oversampling to handle class imbalance
- Cost-sensitive threshold tuning (F2 score optimization prioritizes recall)
- Stratified 5-fold cross-validation for reliable evaluation
- Structured logging and reproducible experiments (SEED=42)

## Pipeline Architecture

```
creditcard.csv  (284,807 transactions)
      │
      ▼
Data Validation       ← schema checks, null checks, fraud rate validation
      │
      ▼
Feature Engineering   ← log_amount, hour, is_small_amount, is_large_amount + V1-V28
      │
      ▼
SMOTE Balancing       ← synthetic minority oversampling (0.172% → 50/50)
      │
      ▼
Model Training        ← Random Forest + XGBoost, stratified 5-fold CV
      │
      ▼
Threshold Tuning      ← F2-score optimization (recall > precision)
      │
      ▼
Automated Reporting   ← JSON metrics + Precision-Recall curves + Confusion matrix
```

## Features

| Feature | Type | Description |
|---------|------|-------------|
| `V1`–`V28` | PCA | Anonymised transaction features (PCA by dataset authors) |
| `log_amount` | Engineered | `log(Amount + 1)` — dampens high-value outliers |
| `hour` | Engineered | Hour of day derived from `Time` field |
| `is_small_amount` | Engineered | Amount < $1 — known card-probing pattern |
| `is_large_amount` | Engineered | Amount > $500 — high-value transaction flag |

## Project Structure

```
FinancialFraudDetection/
├── fraud_detection.py      # Main pipeline
├── requirements.txt        # Dependencies
├── data/                   # Place creditcard.csv here
├── models/                 # Saved model artifacts
├── notebooks/              # Exploratory analysis notebooks
├── outputs/
│   ├── pipeline.log             # Structured run logs
│   ├── performance_report.json  # Model metrics (JSON)
│   └── model_performance.png    # PR curves + confusion matrix
└── README.md
```

## Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

**Option A — Kaggle CLI (recommended)**
```bash
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/
```

**Option B — Manual**
1. Go to [kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv`
3. Place it in the `data/` directory

### 3. Run the pipeline

```bash
python fraud_detection.py
```

**Outputs**
```
outputs/pipeline.log             — timestamped logs for every run
outputs/performance_report.json  — full metrics in JSON
outputs/model_performance.png    — PR curves and confusion matrix
```

## Key Engineering Decisions

### Why SMOTE over class weights?
Class weights adjust the loss function but still train on imbalanced data. SMOTE generates synthetic minority examples, giving the model more fraud patterns to learn from. Applied after the train/test conceptual split so the test set remains the real distribution.

### Why F2 score for threshold tuning?
The default 0.5 threshold optimizes accuracy. F2 weights recall twice as heavily as precision — appropriate for fraud detection where missing real fraud (false negative) costs far more than flagging a legitimate transaction (false positive).

### Why XGBoost over Random Forest?
XGBoost handles class imbalance natively via `scale_pos_weight`, trains faster, and generally outperforms Random Forest on tabular data. Both are included for comparison. Random Forest serves as a strong interpretable baseline.

### Why log-transform Amount?
Transaction amounts are right-skewed with extreme outliers. `log(Amount + 1)` compresses the scale and prevents a handful of large transactions from dominating the model.

### Why structured logging?
Every run produces a timestamped log file — reproducible experiments, debuggable under on-call conditions, and an audit trail for model performance over time.

## Author

**Sanchita Thayagaran**
M.S. Computer Science, UMass Amherst (May 2026)
[linkedin.com/in/sanchitathayagaran](https://linkedin.com/in/sanchitathayagaran)
[sanchita-thayagaran.github.io](https://sanchita-thayagaran.github.io)
