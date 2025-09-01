import os
import io
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify
import requests

# ---------------------------
# CONFIG
# ---------------------------
MODEL_PATH = "best_model.pth"
DRIVE_FILE_ID ="1TVo2YX4Nb5yMHaaPbZkGBHSAs_EemV_n"  # 🔹 replace with your Google Drive file ID
NUM_CLASSES = 38

# ---------------------------
# DOWNLOAD MODEL FROM GOOGLE DRIVE
# ---------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"
        response = requests.get(url)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("Download complete!")

download_model()

# ---------------------------
# LOAD MODEL
# ---------------------------
device = torch.device("cpu")

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ---------------------------
# TRANSFORMS
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Example class names (replace with your actual dataset class labels)
CLASS_NAMES = [f"Class_{i}" for i in range(NUM_CLASSES)]

# ---------------------------
# FLASK APP
# ---------------------------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Plant Disease Detection API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        # Open image
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)

        # Prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()

        class_name = CLASS_NAMES[predicted.item()]

        return jsonify({
            "class": class_name,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
