"""
Multimodal VGG16 Classifier for Reactive Follicular Hyperplasia and Follicular Lymphoma
-----------------------------------------------------------------------------------------
Author: Lucas Lacerda de Souza

Description:
    This script implements a multimodal deep learning pipeline that integrates
    histopathological image patches, clinicopathologic, and nuclear morphometric features
    to classify lymphoid lesions into Reactive Follicular Hyperplasia (RFH) and Follicular Lymphoma (FL).

    The model uses VGG16 for feature extraction from image patches, combines it with
    a clinical/nuclear feature MLP, and evaluates performance at both patch-level and patient-level
    using ROC AUC, calibration curves, confusion matrices, and bootstrapped confidence intervals.

Dependencies:
    torch>=2.1.0
    torchvision>=0.16.0
    pandas>=2.0.0
    numpy>=1.24.0
    matplotlib>=3.8.0
    seaborn>=0.13.0
    scikit-learn>=1.3.0
    pillow>=10.0.0
    tqdm>=4.66.0
    openpyxl>=3.1.0
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve, brier_score_loss
)
from sklearn.calibration import calibration_curve
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dataset Definition
# ---------------------------------------------------------------------------
class MultimodalDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        if "Classe" not in dataframe.columns or "CaseID" not in dataframe.columns:
            raise ValueError("Excel file must contain 'Classe' and 'CaseID' columns.")

        self.image_dir = image_dir
        self.transform = transform
        self.items = []
        self.clinical_cols = [c for c in dataframe.columns if c not in ["Classe", "CaseID"]]

        for _, row in dataframe.iterrows():
            class_dir = os.path.join(image_dir, str(row["Classe"]))
            case_dir = os.path.join(class_dir, str(row["CaseID"]))
            if not os.path.isdir(case_dir):
                continue
            clinical_vec = [float(row[c]) for c in self.clinical_cols]
            for root, _, files in os.walk(case_dir):
                for f in files:
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.items.append({
                            "patch_path": os.path.join(root, f),
                            "clinical": clinical_vec,
                            "label": int(row["Classe"]),
                            "case_id": str(row["CaseID"])
                        })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sample = self.items[idx]
        image = Image.open(sample["patch_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        clinical = torch.tensor(sample["clinical"], dtype=torch.float32)
        label = torch.tensor(sample["label"], dtype=torch.long)
        return image, clinical, label, sample["case_id"]


# ---------------------------------------------------------------------------
# Multimodal VGG16 Model Definition
# ---------------------------------------------------------------------------
class MultimodalVGG16(nn.Module):
    def __init__(self, clinical_input_dim, num_classes=2):
        super().__init__()
        backbone = vgg16(weights=VGG16_Weights.DEFAULT)
        backbone.classifier = nn.Identity()  # Remove original classifier
        self.backbone = backbone.features
        self.avgpool = backbone.avgpool
        self.flatten = nn.Flatten()
        self.backbone_feature_dim = 512 * 7 * 7  # After VGG16 feature extractor

        self.dropout = nn.Dropout(0.5)

        self.clinical_net = nn.Sequential(
            nn.Linear(clinical_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.backbone_feature_dim + 32, 512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, clinical_data):
        x = self.backbone(image)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)

        clinical_features = self.clinical_net(clinical_data)
        combined = torch.cat((x, clinical_features), dim=1)

        return self.classifier(combined)


# ---------------------------------------------------------------------------
# Patient-Level Evaluation and Visualization
# ---------------------------------------------------------------------------
def patient_level_report(df, output_dir=None, n_boot=1000, alpha=0.95, ece_bins=10):
    df_agg = df.groupby("patient_id").agg({"y_true": "max", "y_prob": "mean"}).reset_index()
    y_true = df_agg["y_true"].values
    y_prob = df_agg["y_prob"].values

    auc = roc_auc_score(y_true, y_prob)
    auc_ci = bootstrap((y_true, y_prob), lambda yt, yp: roc_auc_score(yt, yp),
                       n_resamples=n_boot, confidence_level=alpha, paired=True, random_state=42)
    auc_low, auc_high = auc_ci.confidence_interval

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=ece_bins)
    ece = np.mean(np.abs(frac_pos - mean_pred))
    brier = brier_score_loss(y_true, y_prob)

    thresholds = np.linspace(0, 1, 101)
    sens_list, spec_list = [], []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        sens_list.append(sens)
        spec_list.append(spec)

    best_threshold = thresholds[np.argmax(np.array(sens_list) + np.array(spec_list) - 1)]
    final_preds = (y_prob >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, final_preds).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)

    # Visualization
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.savefig(f"{output_dir}/roc.png")
        plt.close()

        plt.figure()
        plt.plot(mean_pred, frac_pos, marker='o', label='Model')
        plt.plot([0, 1], [0, 1], 'k--', label='Ideal')
        plt.xlabel("Predicted Probability")
        plt.ylabel("Observed Proportion")
        plt.title("Calibration Curve")
        plt.legend()
        plt.savefig(f"{output_dir}/calibration_curve.png")
        plt.close()

        cm = confusion_matrix(y_true, final_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.savefig(f"{output_dir}/confusion_matrix.png")
        plt.close()

    return {
        "auc": auc,
        "auc_ci_low": auc_low,
        "auc_ci_high": auc_high,
        "ece": ece,
        "brier": brier,
        "sensitivity": tp / (tp + fn + 1e-9),
        "specificity": tn / (tn + fp + 1e-9),
        "accuracy": acc
    }


# ---------------------------------------------------------------------------
# Main Training and Evaluation Routine
# ---------------------------------------------------------------------------
def main():
    train_dir = "data/train"
    val_dir = "data/val"
    test_dir = "data/test"
    results_dir = "results/vgg16_multimodal"
    os.makedirs(results_dir, exist_ok=True)

    train_df = pd.read_excel(os.path.join(train_dir, "clinical_data_train.xlsx"))
    val_df = pd.read_excel(os.path.join(val_dir, "clinical_data_val.xlsx"))
    test_df = pd.read_excel(os.path.join(test_dir, "clinical_data_test.xlsx"))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = MultimodalDataset(train_df, train_dir, transform)
    test_dataset = MultimodalDataset(test_df, test_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)

    clinical_input_dim = len(train_dataset.clinical_cols)
    model = MultimodalVGG16(clinical_input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Training
    model.train()
    for images, clinical, labels, _ in tqdm(train_loader, desc="Training"):
        images, clinical, labels = images.to(device), clinical.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images, clinical), labels)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    all_labels, all_probs, all_ids = [], [], []
    with torch.no_grad():
        for images, clinical, labels, ids in tqdm(test_loader, desc="Testing"):
            images, clinical = images.to(device), clinical.to(device)
            probs = torch.softmax(model(images, clinical), dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_ids.extend(ids)

    df = pd.DataFrame({"patient_id": all_ids, "y_true": all_labels, "y_prob": all_probs})
    metrics = patient_level_report(df, output_dir=results_dir)

    print(f"\nPatient-level AUC: {metrics['auc']:.3f} (95% CI: {metrics['auc_ci_low']:.3f} - {metrics['auc_ci_high']:.3f})")
    print("Results and plots saved.")


if __name__ == "__main__":
    main()
