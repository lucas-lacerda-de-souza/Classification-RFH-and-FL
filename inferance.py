import os
import argparse
import torch
import pandas as pd
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# =========================
# Argument parsing
# =========================
parser = argparse.ArgumentParser(description="RFH vs FL inference using ResNet18")
parser.add_argument("--input_dir", type=str, required=True, help="Directory with input images")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save predictions")
parser.add_argument("--weights", type=str, default="weights/resnet18_rfh_fl.pth", help="Model weights")
parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
args = parser.parse_args()

# =========================
# Reproducibility
# =========================
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

# =========================
# Model loading
# =========================
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

state_dict = torch.load(args.weights, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# =========================
# Preprocessing
# =========================
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# Inference function
# =========================
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    return pred_class, probs.squeeze().cpu().numpy()

# =========================
# Inference loop
# =========================
os.makedirs(args.output_dir, exist_ok=True)

results = []

class_map = {0: "RFH", 1: "FL"}

image_files = [
    f for f in os.listdir(args.input_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

for fname in tqdm(image_files, desc="Running inference"):
    img_path = os.path.join(args.input_dir, fname)
    pred, prob = predict_image(img_path)

    results.append({
        "image": fname,
        "prediction": class_map[pred],
        "prob_RFH": float(prob[0]),
        "prob_FL": float(prob[1])
    })

# =========================
# Save results
# =========================
df = pd.DataFrame(results)
output_csv = os.path.join(args.output_dir, "predictions.csv")
df.to_csv(output_csv, index=False)

print(f"\nInference completed.")
print(f"Results saved to: {output_csv}")
