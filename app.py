import torch
import torch.nn as nn
import torchvision.models as models
import requests
import os

MODEL_PATH = "best_model.pth"
DRIVE_FILE_ID = "1TVo2YX4Nb5yMHaaPbZkGBHSAs_EemV_n"  # 🔹 replace with your file ID

# Function to download model from Google Drive
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"
        response = requests.get(url)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("Download complete!")

# Download before loading
download_model()

# Load ResNet18 with 38 output classes
model = models.resnet18(pretrained=False)
num_classes = 38
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()
