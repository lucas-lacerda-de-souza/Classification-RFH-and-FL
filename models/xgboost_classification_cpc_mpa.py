"""
XGBoost + SHAP Analysis Script
------------------------------
Author: Lucas Lacerda de Souza

Description:
    This script trains an XGBoost model to differentiate two diagnostic classes:
    Reactive Follicular Hyperplasia (RFH) and Follicular Lymphoma (FL).
    It evaluates model performance on the test set using accuracy, AUC,
    F1-score, precision, and recall. The script also computes SHAP values
    to provide interpretability insights into feature contributions.

Dependencies:
    pip install xgboost shap pandas numpy matplotlib scikit-learn openpyxl
"""

import os
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc
)


def load_and_preprocess_data(file_path: str):
    """Load dataset from Excel file and preprocess predictors and target."""
    data = pd.read_excel(file_path)
    data.rename(columns={data.columns[0]: "Classes"}, inplace=True)
    predictors = [col for col in data.columns if col != "Classes"]

    # Convert predictors to numeric and handle missing values
    data[predictors] = data[predictors].apply(pd.to_numeric, errors='coerce')
    data = data.dropna()

    # Encode class labels
    encoder = LabelEncoder()
    data["Classes"] = encoder.fit_transform(data["Classes"])

    X = data[predictors].values
    y = data["Classes"].values

    return X, y, predictors, encoder


def train_xgboost_model(X_train, y_train):
    """Train an XGBoost binary classification model."""
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=100,
        eta=0.1,
        max_depth=6,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance using standard classification metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred)
    }

    print("\nModel Performance on Test Set")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"AUC:       {metrics['auc']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")

    return y_prob, metrics


def plot_roc_curve(y_test, y_prob, output_path):
    """Plot and save the ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def compute_and_plot_shap(model, X_train, predictors, output_path):
    """Compute and visualize SHAP values."""
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    shap.summary_plot(shap_values, X_train, feature_names=predictors, show=False)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    # ----------------------------------------------------------------------
    # 1. File paths and directories
    # ----------------------------------------------------------------------
    data_path = os.path.join("data", "XGBoost.xlsx")  # Expected structure: ./data/XGBoost.xlsx
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------------------------------
    # 2. Data loading and preprocessing
    # ----------------------------------------------------------------------
    print("Loading and preprocessing dataset...")
    X, y, predictors, encoder = load_and_preprocess_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123, stratify=y
    )

    # ----------------------------------------------------------------------
    # 3. Model training
    # ----------------------------------------------------------------------
    print("Training XGBoost model...")
    model = train_xgboost_model(X_train, y_train)

    # ----------------------------------------------------------------------
    # 4. Model evaluation
    # ----------------------------------------------------------------------
    print("Evaluating model performance...")
    y_prob, metrics = evaluate_model(model, X_test, y_test)

    roc_output = os.path.join(output_dir, "ROC_curve.png")
    plot_roc_curve(y_test, y_prob, roc_output)

    # ----------------------------------------------------------------------
    # 5. SHAP explainability
    # ----------------------------------------------------------------------
    print("Computing SHAP values...")
    shap_output = os.path.join(output_dir, "SHAP_summary.png")
    compute_and_plot_shap(model, X_train, predictors, shap_output)

    # ----------------------------------------------------------------------
    # 6. Summary
    # ----------------------------------------------------------------------
    print(f"\nResults saved to directory: {os.path.abspath(output_dir)}")
    print("Execution completed successfully.")


if __name__ == "__main__":
    main()
