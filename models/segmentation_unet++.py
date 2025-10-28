"""
U-Net++ with Attention for Image Segmentation
---------------------------------------------------------------------------------------
Author: Lucas Lacerda de Souza

Description:
    This script implements U-Net++ with attention gates for image segmentation.
    It includes training, evaluation, and inference functions, and computes
    detailed metrics such as IoU, Dice, F1, and Hausdorff distance.
    It supports mixed-precision training (AMP), CUDA acceleration, and
    multi-GPU execution via DataParallel.

Dependencies:
    - torch>=2.1.0
    - torchvision>=0.16.0
    - numpy>=1.24.0
    - pandas>=2.0.0
    - matplotlib>=3.8.0
    - tqdm>=4.66.0
    - pillow>=10.0.0
    - scipy>=1.11.0

"""

import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import directed_hausdorff


torch.backends.cudnn.benchmark = True  # Enable CUDA convolution auto-tuning


# ==========================================================
# DATASET
# ==========================================================
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=256):
        self.images = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir)
                              if f.endswith((".png", ".jpg", ".jpeg"))])
        self.masks = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir)
                             if f.endswith((".png", ".jpg", ".jpeg"))])
        self.transform_img = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
        self.transform_mask = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")
        return self.transform_img(img), self.transform_mask(mask)


# ==========================================================
# BUILDING BLOCKS
# ==========================================================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, g_channels, x_channels):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Conv2d(g_channels, x_channels, kernel_size=1)
        self.W_x = nn.Conv2d(x_channels, x_channels, kernel_size=1)
        self.psi = nn.Sequential(
            nn.Conv2d(x_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if g1.shape[2:] != x1.shape[2:]:
            g1 = torch.nn.functional.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=True)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        if psi.shape[2:] != x.shape[2:]:
            psi = torch.nn.functional.interpolate(psi, size=x.shape[2:], mode="bilinear", align_corners=True)

        return x * psi


# ==========================================================
# U-NET++ WITH ATTENTION
# ==========================================================
class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, filters=[32, 64, 128, 256, 512]):
        super(UNetPlusPlus, self).__init__()
        self.conv1_0 = ConvBlock(in_channels, filters[0])
        self.pool1 = nn.MaxPool2d(2)
        self.conv2_0 = ConvBlock(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(2)
        self.conv3_0 = ConvBlock(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(2)
        self.conv4_0 = ConvBlock(filters[2], filters[3])
        self.pool4 = nn.MaxPool2d(2)
        self.conv5_0 = ConvBlock(filters[3], filters[4])

        self.att2 = AttentionBlock(filters[1], filters[0])
        self.att3 = AttentionBlock(filters[2], filters[1])
        self.att4 = AttentionBlock(filters[3], filters[2])
        self.att5 = AttentionBlock(filters[4], filters[3])

        self.conv1_1 = ConvBlock(filters[0] * 2, filters[0])
        self.conv2_1 = ConvBlock(filters[1] * 2, filters[1])
        self.conv3_1 = ConvBlock(filters[2] * 2, filters[2])
        self.conv4_1 = ConvBlock(filters[3] * 2, filters[3])

        self.conv1_2 = ConvBlock(filters[0] * 2, filters[0])
        self.conv2_2 = ConvBlock(filters[1] * 2, filters[1])
        self.conv3_2 = ConvBlock(filters[2] * 2, filters[2])

        self.conv1_3 = ConvBlock(filters[0] * 2, filters[0])
        self.conv2_3 = ConvBlock(filters[1] * 2, filters[1])

        self.conv1_4 = ConvBlock(filters[0] * 2, filters[0])
        self.final = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv1_0(x)
        x1_0 = self.conv2_0(self.pool1(x0_0))
        x2_0 = self.conv3_0(self.pool2(x1_0))
        x3_0 = self.conv4_0(self.pool3(x2_0))
        x4_0 = self.conv5_0(self.pool4(x3_0))

        x0_1 = self.conv1_1(torch.cat([x0_0, self.att2(x1_0, x0_0)], 1))
        x1_1 = self.conv2_1(torch.cat([x1_0, self.att3(x2_0, x1_0)], 1))
        x2_1 = self.conv3_1(torch.cat([x2_0, self.att4(x3_0, x2_0)], 1))
        x3_1 = self.conv4_1(torch.cat([x3_0, self.att5(x4_0, x3_0)], 1))

        x0_2 = self.conv1_2(torch.cat([x0_0, x0_1], 1))
        x1_2 = self.conv2_2(torch.cat([x1_0, x1_1], 1))
        x2_2 = self.conv3_2(torch.cat([x2_0, x2_1], 1))

        x0_3 = self.conv1_3(torch.cat([x0_0, x0_2], 1))
        x1_3 = self.conv2_3(torch.cat([x1_0, x1_2], 1))

        x0_4 = self.conv1_4(torch.cat([x0_0, x0_3], 1))
        return self.final(x0_4)


# ==========================================================
# METRICS
# ==========================================================
def hausdorff_distance(mask1, mask2):
    pts1 = np.argwhere(mask1 > 0)
    pts2 = np.argwhere(mask2 > 0)
    if len(pts1) == 0 or len(pts2) == 0:
        return np.nan
    return max(directed_hausdorff(pts1, pts2)[0], directed_hausdorff(pts2, pts1)[0])


def avg_surface_distance(mask1, mask2):
    pts1 = np.argwhere(mask1 > 0)
    pts2 = np.argwhere(mask2 > 0)
    if len(pts1) == 0 or len(pts2) == 0:
        return np.nan
    d1 = np.mean([np.min(np.linalg.norm(p - pts2, axis=1)) for p in pts1])
    d2 = np.mean([np.min(np.linalg.norm(p - pts1, axis=1)) for p in pts2])
    return (d1 + d2) / 2


def compute_metrics(outputs, masks):
    preds = torch.sigmoid(outputs) > 0.5
    preds, masks = preds.float().cpu().numpy(), masks.float().cpu().numpy()

    batch_metrics = []
    for p, m in zip(preds, masks):
        p, m = p.squeeze(), m.squeeze()
        tp = np.logical_and(p == 1, m == 1).sum()
        tn = np.logical_and(p == 0, m == 0).sum()
        fp = np.logical_and(p == 1, m == 0).sum()
        fn = np.logical_and(p == 0, m == 1).sum()

        acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        hd = hausdorff_distance(p, m)
        asd = avg_surface_distance(p, m)

        batch_metrics.append([acc, prec, rec, f1, iou, dice, hd, asd])
    return np.nanmean(batch_metrics, axis=0)


# ==========================================================
# TRAINING PIPELINE
# ==========================================================
def train_model(images_dir, masks_dir, results_dir="results", epochs=50, batch_size=4, img_size=256, device="cuda"):
    os.makedirs(results_dir, exist_ok=True)
    dataset = SegmentationDataset(images_dir, masks_dir, img_size=img_size)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=os.cpu_count(), pin_memory=True)

    model = UNetPlusPlus().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_metrics = []

        for images, masks in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            all_metrics.append(compute_metrics(outputs, masks))

        avg_loss = running_loss / len(loader)
        avg_metrics = np.nanmean(all_metrics, axis=0)
        history.append([epoch+1, avg_loss] + avg_metrics.tolist())

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {avg_metrics[0]:.4f} | "
              f"Prec: {avg_metrics[1]:.4f} | Rec: {avg_metrics[2]:.4f} | F1: {avg_metrics[3]:.4f} | "
              f"IoU: {avg_metrics[4]:.4f} | Dice: {avg_metrics[5]:.4f} | HD: {avg_metrics[6]:.2f} | ASD: {avg_metrics[7]:.2f}")

    df = pd.DataFrame(history, columns=["Epoch","Loss","Acc","Prec","Rec","F1","IoU","Dice","Hausdorff","ASD"])
    df.to_csv(os.path.join(results_dir,"metrics_history.csv"), index=False)

    plt.figure(figsize=(12,5))
    plt.subplot(1,3,1); plt.plot(df["Epoch"], df["Loss"]); plt.title("Loss")
    plt.subplot(1,3,2); plt.plot(df["Epoch"], df["IoU"]); plt.title("IoU")
    plt.subplot(1,3,3); plt.plot(df["Epoch"], df["Dice"]); plt.title("Dice")
    plt.tight_layout(); plt.savefig(os.path.join(results_dir,"training_curves.png")); plt.close()

    model_path = os.path.join(results_dir, "unetpp_attention.pth")
    torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), model_path)
    print(f"Model saved at: {model_path}")
    return model


# ==========================================================
# INFERENCE PIPELINE
# ==========================================================
def infer_and_save(model, input_dir, output_dir, img_size=256, save_size=299, device="cuda"):
    model.eval()
    transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])

    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.endswith((".png", ".jpg", ".jpeg")):
                continue
            img_path = os.path.join(root, file)
            rel_path = os.path.relpath(img_path, input_dir)
            save_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            img = Image.open(img_path).convert("RGB")
            tensor_img = transform(img).unsqueeze(0).to(device, non_blocking=True)

            with torch.no_grad(), torch.cuda.amp.autocast():
                output = model(tensor_img)
                pred = torch.sigmoid(output).cpu().squeeze().numpy()
                mask = (pred > 0.5).astype(np.uint8) * 255

                mask_img = Image.fromarray(mask).resize((save_size, save_size), Image.NEAREST)
                mask_img.save(save_path)

            print(f"Mask saved at: {save_path}")


# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_images_dir = "./train_images"
    train_masks_dir  = "./train_masks"
    new_patches_dir  = "./test_images"
    pred_masks_dir   = "./predicted_masks"

    model = train_model(train_images_dir, train_masks_dir, results_dir="results",
                        epochs=50, batch_size=2, img_size=256, device=device)

    model_infer = UNetPlusPlus().to(device)
    model_infer.load_state_dict(torch.load("results/unetpp_attention.pth", map_location=device))
    if torch.cuda.device_count() > 1:
        model_infer = nn.DataParallel(model_infer)

    infer_and_save(model_infer, new_patches_dir, pred_masks_dir,
                   img_size=256, save_size=299, device=device)
