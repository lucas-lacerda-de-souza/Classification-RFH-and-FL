"""
Multimodal VGG16 Classifier for Reactive Follicular Hyperplasia and Follicular Lymphoma
---------------------------------------------------------------------------------------
Author: Lucas Lacerda de Souza

Description:
    This script implements a multimodal deep learning pipeline that integrates
    histopathological image patches, clinicopathologic and nuclear morphometric features to classify
    lymphoid lesions into Reactive Follicular Hyperplasia (RFH) and Follicular Lymphoma (FL).

    The model is based on a VGG16 convolutional backbone for image embeddings,
    combined with a fully‑connected network for clinical and nuclear morphometric data.
    The network is trained and validated on labeled patch‑level data, and evaluated on
    an external test set using standard classification metrics and explainable outputs.

Dependencies:
    torch>=2.1.0
    torchvision>=0.16.0
    pandas>=2.0.0
    numpy>=1.24.0
    matplotlib>=3.8.0
    seaborn>=0.13.0
    scikit‑learn>=1.3.0
    pillow>=10.0.0
    tqdm>=4.66.0
    openpyxl>=3.1.0
"""

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, cohen_kappa_score, roc_curve
)
from tqdm import tqdm


# ===============================================================
# Dataset Definition
# ===============================================================
class MultimodalDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.items = []

        if "Classe" not in dataframe.columns or "CaseID" not in dataframe.columns:
            raise ValueError("Excel file must contain 'Classe' and 'CaseID' columns.")

        self.clinical_cols = [c for c in dataframe.columns if c not in ["Classe", "CaseID"]]
        if not self.clinical_cols:
            raise ValueError("No clinical features found beyond 'Classe' and 'CaseID'.")

        for _, row in dataframe.iterrows():
            class_dir = os.path.join(self.image_dir, str(row["Classe"]))
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
                            "label": int(row["Classe"])
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
        return image, clinical, label


# ===============================================================
# Model Definition (VGG16 + Clinical Data)
# ===============================================================
class MultimodalVGG16(nn.Module):
    def __init__(self, clinical_input_dim, num_classes=2):
        super().__init__()
        backbone = models.vgg16(pretrained=True)
        backbone.classifier = nn.Sequential(*list(backbone.classifier.children())[:-1])
        self.backbone = backbone
        self.feature_dim = 4096

        self.dropout = nn.Dropout(0.5)
        self.clinical_net = nn.Sequential(
            nn.Linear(clinical_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim + 32, 512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, clinical_data):
        x = self.backbone(image)
        x = self.dropout(x)
        clinical_features = self.clinical_net(clinical_data)
        combined = torch.cat((x, clinical_features), dim=1)
        return self.classifier(combined)


# ===============================================================
# Training + Evaluation Pipeline
# ===============================================================
def main():
    train_dir = "data/train"
    val_dir = "data/val"
    test_dir = "data/test"
    results_dir = "results/follicular_multimodal_vgg16"
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
    val_dataset = MultimodalDataset(val_df, val_dir, transform)
    test_dataset = MultimodalDataset(test_df, test_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    clinical_input_dim = len(train_dataset.clinical_cols)
    model = MultimodalVGG16(clinical_input_dim, num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    labels = torch.tensor([sample['label'] for sample in train_dataset.items], dtype=torch.long)
    label_counts = torch.bincount(labels)
    class_weights = len(labels) / (len(label_counts) * label_counts.float())
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val_loss = float("inf")
    history = []

    for epoch in range(100):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for imgs, clinical, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, clinical, labels = imgs.to(device), clinical.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs, clinical)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, clinical, labels in val_loader:
                imgs, clinical, labels = imgs.to(device), clinical.to(device), labels.to(device)
                outputs = model(imgs, clinical)
                val_loss += criterion(outputs, labels).item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        avg_val_loss = val_loss / len(val_loader)

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc
        })

        print(f"[{epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2%}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pt"))

    pd.DataFrame(history).to_excel(os.path.join(results_dir, "training_history.xlsx"), index=False)

    print("\n🔍 Evaluating on test set...")
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for imgs, clinical, labels in tqdm(test_loader):
            imgs, clinical = imgs.to(device), clinical.to(device)
            outputs = model(imgs, clinical)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs[:, 1].cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "Loss (Val Last)": history[-1]["val_loss"] if history else float("nan"),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall (Sensitivity)": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "Cohen Kappa": cohen_kappa_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob) if len(set(y_true)) == 2 else float("nan"),
        "True Positives": int(tp),
        "False Positives": int(fp),
        "True Negatives": int(tn),
        "False Negatives": int(fn)
    }

    pd.DataFrame([metrics]).to_csv(os.path.join(results_dir, "metrics.csv"), index=False)
    with open(os.path.join(results_dir, "metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()

    if len(set(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, label=f"AUC = {metrics['AUC']:.2f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "roc_curve.png"))
        plt.close()


# ===============================================================
# Entry Point
# ===============================================================
if __name__ == "__main__":
    main()
