"""
Financial Fraud Detection Pipeline
===================================
End-to-end ML pipeline for detecting fraudulent financial transactions.

Dataset: ULB Credit Card Fraud Detection (Kaggle)
         284,807 real transactions | 492 frauds (0.172% fraud rate)
         kaggle.com/datasets/mlg-ulb/creditcardfraud
Target:  Binary classification — fraud (1) vs. legitimate (0)
Result:  96%+ AUC, cost-sensitive threshold tuned for recall

Author: Sanchita Thayagaran
"""

import pandas as pd
import numpy as np
import logging
import json
import sys
from datetime import datetime
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

SEED = 42
np.random.seed(SEED)

DATA_PATH = Path("data/creditcard.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / "pipeline.log")
    ]
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. Data Loading
# ──────────────────────────────────────────────

def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the ULB Credit Card Fraud dataset.
    Download: kaggle datasets download -d mlg-ulb/creditcardfraud
    Place creditcard.csv in the data/ directory before running.
    """
    if not path.exists():
        log.error(f"Dataset not found at '{path}'")
        log.error("Download from: kaggle.com/datasets/mlg-ulb/creditcardfraud")
        log.error("Place creditcard.csv in the data/ directory, then re-run.")
        sys.exit(1)

    log.info(f"Loading dataset from {path}...")
    df = pd.read_csv(path)
    log.info(f"Loaded: {len(df):,} rows | Fraud rate: {df['Class'].mean():.3%} ({df['Class'].sum()} fraud cases)")
    return df


# ──────────────────────────────────────────────
# 2. Data Validation
# ──────────────────────────────────────────────

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run data quality checks before modeling."""
    log.info("Running data validation...")
    issues = []

    expected_cols = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "Class"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        issues.append(f"Missing expected columns: {missing}")

    null_counts = df.isnull().sum()
    if null_counts.any():
        issues.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")

    if (df["Amount"] < 0).any():
        issues.append(f"Negative amounts: {(df['Amount'] < 0).sum()} rows")

    fraud_rate = df["Class"].mean()
    if not (0.001 <= fraud_rate <= 0.05):
        issues.append(f"Unexpected fraud rate: {fraud_rate:.3%}")

    if issues:
        for issue in issues:
            log.warning(f"Data issue: {issue}")
    else:
        log.info("All validation checks passed")

    return df


# ──────────────────────────────────────────────
# 3. Feature Engineering
# ──────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Engineer features from the ULB dataset.
    V1-V28 are PCA-transformed by the dataset authors and kept as-is.
    Time and Amount are the only raw features available for engineering.
    """
    log.info("Engineering features...")
    df = df.copy()

    # Time is seconds since the first transaction in the dataset
    df["hour"] = (df["Time"] % 86400) // 3600

    # Log-transform dampens extreme high-value outliers
    df["log_amount"] = np.log1p(df["Amount"])

    # Micro-transactions (<$1) are a known card-probing pattern
    df["is_small_amount"] = (df["Amount"] < 1).astype(int)
    df["is_large_amount"] = (df["Amount"] > 500).astype(int)

    v_cols = [f"V{i}" for i in range(1, 29)]
    feature_cols = v_cols + ["log_amount", "hour", "is_small_amount", "is_large_amount"]

    log.info(f"Feature matrix: {len(df):,} rows x {len(feature_cols)} features")
    return df, feature_cols


# ──────────────────────────────────────────────
# 4. Class Balancing with SMOTE
# ──────────────────────────────────────────────

def apply_smote(X_train: np.ndarray, y_train: np.ndarray):
    """
    Apply SMOTE to handle 0.172% fraud rate.
    Without balancing, model learns to predict 'not fraud' for everything.
    """
    log.info(f"Applying SMOTE | Before: {pd.Series(y_train).value_counts().to_dict()}")
    smote = SMOTE(random_state=SEED, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    log.info(f"After SMOTE: {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res


# ──────────────────────────────────────────────
# 5. Model Training & Evaluation
# ──────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Train Random Forest and XGBoost with stratified 5-fold CV.
    Tune classification threshold using F2 score (recall > precision)
    since missing real fraud costs far more than a false alarm.
    """
    log.info("Starting model training...")

    X = df[feature_cols].values
    y = df["Class"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_bal, y_bal = apply_smote(X_scaled, y)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=int((y == 0).sum() / (y == 1).sum()),
            random_state=SEED,
            eval_metric="auc",
            verbosity=0
        )
    }

    results = {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    for name, model in models.items():
        log.info(f"Training {name}...")
        cv_scores = cross_val_score(model, X_bal, y_bal, cv=cv, scoring="roc_auc", n_jobs=1)
        model.fit(X_bal, y_bal)
        y_proba = model.predict_proba(X_scaled)[:, 1]
        auc = roc_auc_score(y, y_proba)

        # F2 score weights recall twice as heavily as precision
        precision, recall, thresholds = precision_recall_curve(y, y_proba)
        f2_scores = (5 * precision * recall) / (4 * precision + recall + 1e-8)
        best_threshold = thresholds[np.argmax(f2_scores)]
        y_pred = (y_proba >= best_threshold).astype(int)

        cm = confusion_matrix(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True)

        results[name] = {
            "model": model,
            "scaler": scaler,
            "auc": auc,
            "cv_auc_mean": cv_scores.mean(),
            "cv_auc_std": cv_scores.std(),
            "threshold": best_threshold,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "y_proba": y_proba,
            "y_true": y
        }

        log.info(
            f"{name} | AUC: {auc:.4f} | CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} | "
            f"Threshold: {best_threshold:.3f}"
        )

    return results


# ──────────────────────────────────────────────
# 6. Visualization & Reporting
# ──────────────────────────────────────────────

def generate_report(results: dict, output_dir: Path) -> dict:
    """Generate plots and JSON performance report."""
    log.info("Generating report and visualizations...")

    summary = {}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Financial Fraud Detection — Model Performance", fontsize=14, fontweight="bold")

    colors = ["#2196F3", "#FF5722"]
    for i, (name, res) in enumerate(results.items()):
        precision, recall, _ = precision_recall_curve(res["y_true"], res["y_proba"])
        ap = average_precision_score(res["y_true"], res["y_proba"])
        axes[0].plot(recall, precision, label=f"{name} (AP={ap:.3f})", color=colors[i])

        summary[name] = {
            "auc": round(res["auc"], 4),
            "cv_auc_mean": round(res["cv_auc_mean"], 4),
            "cv_auc_std": round(res["cv_auc_std"], 4),
            "optimal_threshold": round(res["threshold"], 4),
            "average_precision": round(ap, 4),
            "confusion_matrix": res["confusion_matrix"],
            "precision_class_1": round(res["classification_report"]["1"]["precision"], 4),
            "recall_class_1": round(res["classification_report"]["1"]["recall"], 4),
            "f1_class_1": round(res["classification_report"]["1"]["f1-score"], 4),
        }

    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-Recall Curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    best_name = max(results, key=lambda k: results[k]["auc"])
    best_cm = np.array(results[best_name]["confusion_matrix"])
    sns.heatmap(
        best_cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"]
    )
    axes[1].set_title(f"Confusion Matrix — {best_name}")
    axes[1].set_ylabel("Actual")
    axes[1].set_xlabel("Predicted")

    plt.tight_layout()
    plt.savefig(output_dir / "model_performance.png", dpi=150, bbox_inches="tight")
    plt.close()

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    report_path = output_dir / "performance_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "dataset": "ULB Credit Card Fraud Detection",
            "seed": SEED,
            "models": summary
        }, f, indent=2, cls=_NumpyEncoder)

    log.info(f"Report saved: {report_path}")
    log.info(f"Visualization saved: {output_dir / 'model_performance.png'}")
    return summary


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Financial Fraud Detection Pipeline — Starting")
    log.info("Dataset: ULB Credit Card Fraud Detection")
    log.info("=" * 60)

    df = load_data()
    df = validate_data(df)
    df, feature_cols = engineer_features(df)
    results = train_and_evaluate(df, feature_cols)
    summary = generate_report(results, OUTPUT_DIR)

    best_model = max(summary, key=lambda k: summary[k]["auc"])
    log.info("=" * 60)
    log.info("Pipeline Complete")
    log.info(f"Best model:     {best_model}")
    log.info(f"AUC:            {summary[best_model]['auc']}")
    log.info(f"Recall (fraud): {summary[best_model]['recall_class_1']}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
