from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import joblib
import wikipedia
import requests
import os 


app = Flask(__name__)

# ------------------------------
# Crop Recommendation
# ------------------------------
crop_model = joblib.load("crop_rf_model.joblib")
encoder = joblib.load("crop_label_encoder.joblib")

@app.route("/predict-crop", methods=["POST"])
def predict_crop():
    try:
        data = request.get_json()
        lat = float(data["latitude"])
        lon = float(data["longitude"])
        X = np.array([[lat, lon]])
        probs = crop_model.predict_proba(X)[0]
        top_indices = probs.argsort()[-3:][::-1]
        predictions = []

        for idx in top_indices:
            crop = encoder.inverse_transform([idx])[0]
            confidence = float(probs[idx] * 100)
            try:
                info = wikipedia.summary(crop, sentences=5)
                wiki_url = wikipedia.page(crop).url
            except Exception:
                info = f"No detailed information available for {crop}."
                wiki_url = None
            predictions.append({
                "crop": crop,
                "confidence": confidence,
                "info": info,
                "wiki_url": wiki_url
            })

        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# ------------------------------
# Disease Detection
# ------------------------------
num_classes = 38
disease_model = models.resnet18(pretrained=False)
disease_model.fc = nn.Linear(disease_model.fc.in_features, num_classes)
state_dict = torch.load("best_model.pth", map_location=torch.device("cpu"))
disease_model.load_state_dict(state_dict)
disease_model.eval()
disease_model.cpu()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class_names = [
    "Apple__Apple_scab", "Apple_Black_rot", "Apple_Cedar_apple_rust", "Apple__healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)Powdery_mildew", "Cherry(including_sour)_healthy",
    "Corn_(maize)Cercospora_leaf_spot Gray_leaf_spot", "Corn(maize)Common_rust",
    "Corn_(maize)Northern_Leaf_Blight", "Corn(maize)_healthy",
    "Grape__Black_rot", "Grape_Esca(Black_Measles)", "Grape__Leaf_blight(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange__Haunglongbing(Citrus_greening)",
    "Peach__Bacterial_spot", "Peach__healthy",
    "Pepper,bell_Bacterial_spot", "Pepper,_bell__healthy",
    "Potato__Early_blight", "Potato_Late_blight", "Potato__healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry__Leaf_scorch", "Strawberry__healthy",
    "Tomato__Bacterial_spot", "Tomato_Early_blight", "Tomato__Late_blight",
    "Tomato__Leaf_Mold", "Tomato__Septoria_leaf_spot",
    "Tomato__Spider_mites Two-spotted_spider_mite", "Tomato__Target_Spot",
    "Tomato__Tomato_Yellow_Leaf_Curl_Virus", "Tomato_Tomato_mosaic_virus", "Tomato__healthy"
]

CONFIDENCE_THRESHOLD = 0.60
MARGIN_THRESHOLD = 0.15
GREEN_RATIO_MIN = 0.03

def green_pixel_ratio(pil_image):
    arr = np.array(pil_image).astype(np.int16)
    if arr.ndim != 3 or arr.shape[2] < 3: 
        return 0.0
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    green_mask = (g > r+15) & (g > b+15) & (g>80)
    return float(green_mask.sum() / (r.size+1e-9))

@app.route("/predict-disease", methods=["POST"])
def predict_disease():
    try:
        if "file" not in request.files:
            return jsonify({"error":"No file uploaded"}),400

        file = request.files["file"]
        image = Image.open(file.stream).convert("RGB")
        green_ratio = green_pixel_ratio(image)
        img_tensor = transform(image).unsqueeze(0).cpu()

        with torch.no_grad():
            outputs = disease_model(img_tensor)
            probs = F.softmax(outputs, dim=1)[0]
            topk_vals, topk_idx = torch.topk(probs, 2)
            top1_conf = float(topk_vals[0].item())
            top2_conf = float(topk_vals[1].item())
            margin = top1_conf - top2_conf
            predicted_label = class_names[int(topk_idx[0].item())]

        # Safe Wikipedia fetch
        try:
            disease_info = wikipedia.summary(predicted_label, sentences=5)
            wiki_url = wikipedia.page(predicted_label).url
        except Exception:
            disease_info = "No additional info available for this disease."
            wiki_url = None

        # Dummy medicine & shop suggestions
        suggestions = [
            {"medicine": "Neem oil", "usage": "Spray on affected leaves", "shop": "Local Agro Shop"},
            {"medicine": "Copper fungicide", "usage": "Apply weekly", "shop": "Agro Chemicals Store"},
        ]

        response = {
            "class": predicted_label if top1_conf>CONFIDENCE_THRESHOLD else "Unknown",
            "confidence": round(top1_conf,4),
            "margin": round(margin,4),
            "green_ratio": round(green_ratio,4),
            "info": disease_info,
            "wiki_url": wiki_url,
            "suggestions": suggestions
        }

        return jsonify(response), 200

    except Exception as e:
        # Always return JSON
        return jsonify({"error": f"Server error: {str(e)}"}), 500

    try:
        if "file" not in request.files:
            return jsonify({"error":"No file uploaded"}),400

        file = request.files["file"]
        image = Image.open(file.stream).convert("RGB")
        green_ratio = green_pixel_ratio(image)
        img_tensor = transform(image).unsqueeze(0).cpu()

        with torch.no_grad():
            outputs = disease_model(img_tensor)
            probs = F.softmax(outputs, dim=1)[0]
            topk_vals, topk_idx = torch.topk(probs, 2)
            top1_conf = float(topk_vals[0].item())
            top2_conf = float(topk_vals[1].item())
            margin = top1_conf - top2_conf
            predicted_label = class_names[int(topk_idx[0].item())]

        # Safe Wikipedia fetch
        try:
            disease_info = wikipedia.summary(predicted_label, sentences=5)
            wiki_url = wikipedia.page(predicted_label).url
        except Exception:
            disease_info = "No additional info available for this disease."
            wiki_url = None

        # Dummy medicine & shop suggestions (can be replaced by real DB/API)
        suggestions = [
            {"medicine": "Neem oil", "usage": "Spray on affected leaves"},
            {"medicine": "Copper fungicide", "usage": "Apply weekly"},
        ]

        response = {
            "class": predicted_label if top1_conf>CONFIDENCE_THRESHOLD else "Unknown",
            "confidence": round(top1_conf,4),
            "margin": round(margin,4),
            "green_ratio": round(green_ratio,4),
            "info": disease_info,
            "wiki_url": wiki_url,
            "suggestions": suggestions
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# ------------------------------
# Market Value Fetching
# ------------------------------
@app.route("/market", methods=["GET"])
def market_info():
    crop_name = request.args.get("crop", "").strip()
    if not crop_name:
        return jsonify({"error": "Please provide a crop name ?crop=Rice"}), 400

    result = {"crop": crop_name}

    # Wikipedia summary
    try:
        summary = wikipedia.summary(crop_name, sentences=3)
        wiki_url = wikipedia.page(crop_name).url
        result["info"] = summary
        result["wiki_url"] = wiki_url
    except Exception as e:
        result["info"] = f"No Wikipedia info found. ({str(e)})"
        result["wiki_url"] = None

    # Market price fetch
    try:
        url = f"https://pricesapi.datayuge.in/markets?crop={crop_name}"
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            result["market_price"] = data.get("price", "Not available") if isinstance(data, dict) else "Not available"
            result["market_location"] = data.get("market", "Unknown") if isinstance(data, dict) else "Unknown"
        else:
            result["market_price"] = "API error"
            result["market_location"] = "Unknown"
    except Exception as e:
        result["market_price"] = f"Error fetching market price ({str(e)})"
        result["market_location"] = "Unknown"

    return jsonify(result)

# ------------------------------
# Run Server
# ------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # ✅ Railway sets PORT
    app.run(host="0.0.0.0", port=port, debug=False)