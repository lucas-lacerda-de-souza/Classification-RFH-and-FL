"""
XGBoost with Train / Validation / External Validation
------------------------------
Author: Lucas Lacerda de Souza

Description:
    This script trains and evaluates an XGBoost model using all nuclear factors combined
    (multivariate model). It performs internal validation (train/validation split)
    and external validation using a separate dataset. The output includes a unified 
    performance plot comparing Training, Validation, and External Validation sets.

Dependencies:
    - xgboost
    - pandas
    - numpy
    - scikit-learn
    - matplotlib
    - seaborn
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# =========================================================
# Load Datasets
# =========================================================
def load_data(train_path, test_path):
    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path)

    features = [
        "Nucleus: Area",
        "Nucleus: Perimeter",
        "Nucleus: Circularity",
        "Nucleus: Eccentricity",
        "Nucleus: Hematoxylin OD mean"
    ]

    train_df = train_df[["Classe"] + features]
    test_df = test_df[["Classe"] + features]

    return train_df, test_df


# =========================================================
# XGBoost Training
# =========================================================
def train_xgboost(train_df, test_df, results_dir="Results_XGBoost_AllFactors", seed=123):
    os.makedirs(results_dir, exist_ok=True)

    X = train_df.drop(columns=["Classe"])
    y = train_df["Classe"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    X_test = test_df.drop(columns=["Classe"])
    y_test = test_df["Classe"]

    model = XGBClassifier(
        objective="binary:logistic",
        booster="gbtree",
        learning_rate=0.01,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        n_estimators=3000,
        eval_metric="aucpr",
        random_state=seed,
        use_label_encoder=False
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        early_stopping_rounds=100,
        verbose=True
    )

    preds_train = model.predict_proba(X_train)[:, 1]
    preds_val = model.predict_proba(X_val)[:, 1]
    preds_test = model.predict_proba(X_test)[:, 1]

    return model, (X_train, y_train, preds_train), (X_val, y_val, preds_val), (X_test, y_test, preds_test)


# =========================================================
# Evaluation
# =========================================================
def compute_metrics(y_true, y_prob):
    y_pred = (y_prob > 0.5).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1_Score": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_prob)
    }


# =========================================================
# Custom Metrics (Editable Section)
# =========================================================
def custom_metrics():
    return pd.DataFrame({
        "Set": ["Training", "Validation", "External Validation"],
        "Accuracy":  [0.926, 0.776, 0.817],
        "Precision": [0.937, 0.778, 0.834],
        "Recall":    [0.949, 0.767, 0.824],
        "F1_Score":  [0.926, 0.742, 0.819],
        "AUC":       [0.934, 0.737, 0.847]
    })


# =========================================================
# Visualization
# =========================================================
def plot_metrics(metrics_df, results_dir="Results_XGBoost_AllFactors"):
    metrics_long = metrics_df.melt(id_vars="Set", var_name="Metric", value_name="Value")
    metrics_long["Set"] = pd.Categorical(metrics_long["Set"],
                                         categories=["Training", "Validation", "External Validation"],
                                         ordered=True)

    sns.set(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=metrics_long, x="Metric", y="Value", hue="Set",
                     palette=["#2166AC", "#67A9CF", "#D1E5F0"])
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", label_type="edge", padding=3)
    plt.ylim(0, 1.05)
    plt.title("XGBoost Performance (All Nuclear Factors)\nTraining | Validation | External Validation",
              fontsize=15, weight="bold")
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.legend(title="Dataset", loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
    plt.tight_layout()

    save_path = os.path.join(results_dir, "all_datasets_metrics_customizable.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved at: {save_path}")


# =========================================================
# Main Execution
# =========================================================
if __name__ == "__main__":
    train_path = "./data/XGBoost_Train.xlsx"
    test_path = "./data/XGBoost_Test.xlsx"
    results_dir = "./Results_XGBoost_AllFactors"

    train_df, test_df = load_data(train_path, test_path)

    model, train_res, val_res, test_res = train_xgboost(train_df, test_df, results_dir=results_dir)

    metrics_train = compute_metrics(train_res[1], train_res[2])
    metrics_val = compute_metrics(val_res[1], val_res[2])
    metrics_test = compute_metrics(test_res[1], test_res[2])

    metrics_df = pd.DataFrame([
        {"Set": "Training", **metrics_train},
        {"Set": "Validation", **metrics_val},
        {"Set": "External Validation", **metrics_test},
    ])

    # Optional: replace computed metrics with manually averaged ones
    # metrics_df = custom_metrics()

    print("\nFinal Metrics:")
    print(metrics_df.round(3))

    metrics_df.to_csv(os.path.join(results_dir, "xgboost_all_metrics.csv"), index=False)
    plot_metrics(metrics_df, results_dir=results_dir)
