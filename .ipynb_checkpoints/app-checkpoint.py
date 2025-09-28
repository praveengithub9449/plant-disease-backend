from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model & encoder
model = joblib.load("crop_rf_model.joblib")  # use your file name
encoder = joblib.load("label_encoder.pkl")   # use your file name

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        lat = float(data["latitude"])
        lon = float(data["longitude"])

        # Input must match training features
        X = np.array([[lat, lon]])
        
        # Get prediction probabilities
        probs = model.predict_proba(X)[0]
        top_indices = probs.argsort()[-3:][::-1]  # Top 3
        top_crops = encoder.inverse_transform(top_indices)

        return jsonify({"predictions": top_crops.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
