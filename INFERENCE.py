import torch
from torchvision import models, transforms
from PIL import Image
import os
import argparse

parser = argparse.ArgumentParser(description="RFH vs FL Inference")
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
args = parser.parse_args()

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('weights/resnet18_rfh_fl.pth', map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_t)
        pred = torch.argmax(output, 1).item()
    return "RFH" if pred == 0 else "FL"

os.makedirs(args.output_dir, exist_ok=True)
for file in os.listdir(args.input_dir):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        label = predict_image(os.path.join(args.input_dir, file))
        print(f"{file}: {label}")
        with open(os.path.join(args.output_dir, 'predictions.txt'), 'a') as f:
            f.write(f"{file}: {label}\n")

