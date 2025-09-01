{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c8a3d7fa-bd95-487f-b0ad-9195d09b17ab",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "E:\\anaconda\\Lib\\site-packages\\torchvision\\models\\_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.\n",
      "  warnings.warn(\n",
      "E:\\anaconda\\Lib\\site-packages\\torchvision\\models\\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.\n",
      "  warnings.warn(msg)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app '__main__'\n",
      " * Debug mode: off\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\n",
      " * Running on all addresses (0.0.0.0)\n",
      " * Running on http://127.0.0.1:5000\n",
      " * Running on http://10.46.113.1:5000\n",
      "Press CTRL+C to quit\n",
      "10.46.113.1 - - [01/Sep/2025 21:38:08] \"GET / HTTP/1.1\" 404 -\n",
      "10.46.113.1 - - [01/Sep/2025 21:38:08] \"GET /favicon.ico HTTP/1.1\" 404 -\n"
     ]
    }
   ],
   "source": [
    "from flask import Flask, request, jsonify\n",
    "import torch\n",
    "from torchvision import transforms, models\n",
    "from PIL import Image\n",
    "\n",
    "app = Flask(__name__)\n",
    "\n",
    "# Load model\n",
    "NUM_CLASSES = 38  # 👈 change to your number of classes\n",
    "model = models.resnet18(pretrained=False)\n",
    "model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)\n",
    "model.load_state_dict(torch.load(\"best_model.pth\", map_location=\"cpu\"))\n",
    "model.eval()\n",
    "\n",
    "# Preprocessing\n",
    "transform = transforms.Compose([\n",
    "    transforms.Resize((224,224)),\n",
    "    transforms.ToTensor(),\n",
    "    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])\n",
    "])\n",
    "\n",
    "@app.route('/predict', methods=['POST'])\n",
    "def predict():\n",
    "    if 'file' not in request.files:\n",
    "        return jsonify({'error': 'No file uploaded'})\n",
    "    \n",
    "    file = request.files['file']\n",
    "    img = Image.open(file.stream).convert(\"RGB\")\n",
    "    img = transform(img).unsqueeze(0)\n",
    "    \n",
    "    with torch.no_grad():\n",
    "        output = model(img)\n",
    "        _, pred = torch.max(output, 1)\n",
    "    \n",
    "    classes =  [\n",
    "        'Apple___Apple_scab',\n",
    "    'Apple___Black_rot',\n",
    "    'Apple___Cedar_apple_rust',\n",
    "    'Apple___healthy',\n",
    "    'Blueberry___healthy',\n",
    "    'Cherry_(including_sour)___Powdery_mildew',\n",
    "    'Cherry_(including_sour)___healthy',\n",
    "    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',\n",
    "    'Corn_(maize)___Common_rust_',\n",
    "    'Corn_(maize)___Northern_Leaf_Blight',\n",
    "    'Corn_(maize)___healthy',\n",
    "    'Grape___Black_rot',\n",
    "    'Grape___Esca_(Black_Measles)',\n",
    "    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',\n",
    "    'Grape___healthy',\n",
    "    'Orange___Haunglongbing_(Citrus_greening)',\n",
    "    'Peach___Bacterial_spot',\n",
    "    'Peach___healthy',\n",
    "    'Pepper,_bell___Bacterial_spot',\n",
    "    'Pepper,_bell___healthy',\n",
    "    'Potato___Early_blight',\n",
    "    'Potato___Late_blight',\n",
    "    'Potato___healthy',\n",
    "    'Raspberry___healthy',\n",
    "    'Soybean___healthy',\n",
    "    'Squash___Powdery_mildew',\n",
    "    'Strawberry___Leaf_scorch',\n",
    "    'Strawberry___healthy',\n",
    "    'Tomato___Bacterial_spot',\n",
    "    'Tomato___Early_blight',\n",
    "    'Tomato___Late_blight',\n",
    "    'Tomato___Leaf_Mold',\n",
    "    'Tomato___Septoria_leaf_spot',\n",
    "    'Tomato___Spider_mites Two-spotted_spider_mite',\n",
    "    'Tomato___Target_Spot',\n",
    "    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',\n",
    "    'Tomato___Tomato_mosaic_virus',\n",
    "    'Tomato___healthy']\n",
    "\n",
    "    return jsonify({\"prediction\": classes[pred.item()]})\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    app.run(host=\"0.0.0.0\", port=5000)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "07ecda4e-e6fc-43e6-a5f8-ea0aa4545d0c",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
