# Financial Fraud Detection Pipeline

An end-to-end machine learning pipeline for detecting fraudulent financial transactions at scale.

## Results

| Model | AUC | CV AUC | False Negatives vs Baseline |
|-------|-----|--------|-----------------------------|
| XGBoost | **96.2%** | 95.8% ± 0.3% | **-22%** |
| Random Forest | 94.7% | 94.1% ± 0.4% | -18% |

## Problem

Financial fraud detection is a classic **severe class imbalance** problem:
- ~1.5% of transactions are fraudulent
- A naive model that predicts "not fraud" every time achieves 98.5% accuracy but catches zero fraud
- **False negatives** (missed fraud) cost far more than false positives (false alarms)

This pipeline solves those challenges with:
- SMOTE oversampling to handle class imbalance
- Cost-sensitive threshold tuning (F2 score optimization prioritizes recall)
- Stratified k-fold cross-validation for reliable evaluation
- Structured logging and reproducible experiments

## Pipeline Stages

```
Raw Data (3M+ transactions)
    │
    ▼
Data Validation          ← null checks, amount validation, fraud rate checks
    │
    ▼
Feature Engineering      ← 15 features: ratios, flags, risk scores
    │
    ▼
SMOTE Balancing          ← synthetic minority oversampling
    │
    ▼
Model Training           ← Random Forest + XGBoost with stratified 5-fold CV
    │
    ▼
Threshold Tuning         ← F2-score optimization (recall > precision)
    │
    ▼
Automated Reporting      ← JSON metrics + Precision-Recall curves + Confusion matrix
```

## Features

| Feature | Description |
|---------|-------------|
| `amount` | Transaction amount |
| `hour_of_day` | Hour of transaction (fraud skews 0-5am) |
| `day_of_week` | Day of week |
| `merchant_category_encoded` | Online, retail, travel, grocery, entertainment |
| `transaction_count_24h` | Number of transactions in last 24 hours |
| `avg_amount_30d` | 30-day average transaction amount |
| `distance_from_home_km` | Distance from cardholder's home |
| `is_international` | International transaction flag |
| `failed_attempts_24h` | Failed transaction attempts in 24h |
| `amount_to_avg_ratio` | Current amount / 30-day average |
| `is_high_frequency` | >5 transactions in 24h |
| `is_large_amount` | Amount > $500 |
| `is_late_night` | Transaction between midnight and 5am |
| `is_weekend` | Weekend transaction flag |
| `risk_score` | Composite risk score |

## Project Structure

```
FinancialFraudDetection/
├── fraud_detection.py      # Main pipeline
├── requirements.txt        # Dependencies
├── data/                   # Raw and processed datasets
├── models/                 # Saved model artifacts
├── notebooks/              # Exploratory analysis notebooks
├── outputs/
│   ├── pipeline.log        # Structured run logs
│   ├── performance_report.json  # Model metrics
│   └── model_performance.png   # Visualizations
└── README.md
```

## Setup & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python fraud_detection.py

# Output
# outputs/pipeline.log          — structured logs for every run
# outputs/performance_report.json — full metrics in JSON
# outputs/model_performance.png   — PR curves and confusion matrix
```

## Key Engineering Decisions

### Why SMOTE over class weights?
Class weights adjust the loss function but still train on imbalanced data. SMOTE generates synthetic minority examples, giving the model more fraud patterns to learn from. Combined with stratified k-fold, this prevents data leakage from the resampling step.

### Why F2 score for threshold tuning?
Standard threshold (0.5) optimizes accuracy. F2 score weights recall twice as heavily as precision — appropriate for fraud detection where missing real fraud (false negative) costs far more than flagging a legitimate transaction (false positive).

### Why XGBoost over Random Forest?
XGBoost handles class imbalance natively via `scale_pos_weight`, trains faster on large datasets, and generally outperforms Random Forest on tabular financial data. Both are included for comparison.

### Why structured logging?
Every pipeline run produces a timestamped log file. This means:
- Reproducible experiments (seed control throughout)
- Debuggable under on-call conditions by engineers who didn't write the code
- Audit trail for model performance over time

## Author

**Sanchita Thayagaran**  
M.S. Computer Science, UMass Amherst (May 2026)  
[linkedin.com/in/sanchitathayagaran](https://linkedin.com/in/sanchitathayagaran)  
[sanchita-thayagaran.github.io](https://sanchita-thayagaran.github.io)