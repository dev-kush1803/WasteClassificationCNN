from flask import Flask, request, render_template
import gdown
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Google Drive File ID of the Model
file_id = "1i1WviV9Q7pe9VGP4l8dn3LpBVkcTfHbC"
model_path = "waste_classifier.keras"

# Function to download the model if not present
def download_model():
    if not os.path.exists(model_path):  # Check if model already exists
        print("Downloading model from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

# Download model before starting app
download_model()

# Load the trained model
model = load_model(model_path)

# Function to preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224)) / 255.0  # Resize & Normalize
    img = np.reshape(img, (1, 224, 224, 3))  # Reshape for model input
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        file_path = "static/uploaded_image.jpg"
        file.save(file_path)

        # Preprocess & Predict
        img = preprocess_image(file_path)
        prediction = model.predict(img)[0]
        result = "♻️ Recyclable Waste" if np.argmax(prediction) == 0 else "🍃 Organic Waste"

        return render_template("index.html", result=result, image_path=file_path)

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
