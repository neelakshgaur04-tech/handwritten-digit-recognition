from pathlib import Path
import base64
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "digit_model.keras"

app = Flask(__name__)
model = None


def load_trained_model():
    """Load the trained model once when the application starts."""
    global model
    if not MODEL_PATH.exists():
        return
    model = tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(data_url: str) -> np.ndarray:
    """Convert a canvas image into the 28x28 MNIST input format."""
    if "," not in data_url:
        raise ValueError("Invalid image data.")

    encoded_image = data_url.split(",", 1)[1]
    image = Image.open(BytesIO(base64.b64decode(encoded_image))).convert("L")
    image = image.resize((28, 28))

    pixels = np.array(image, dtype=np.float32)
    pixels = 255.0 - pixels          # white drawing -> bright digit
    pixels /= 255.0                  # normalize to 0-1

    return pixels.reshape(1, 28, 28)


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/predict")
def predict():
    if model is None:
        return jsonify({
            "error": "Model not found. Run 'python train_model.py' first."
        }), 503

    try:
        payload = request.get_json(silent=True) or {}
        image_data = payload.get("image")

        if not image_data:
            return jsonify({"error": "No drawing was received."}), 400

        image = preprocess_image(image_data)
        probabilities = model.predict(image, verbose=0)[0]

        digit = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities) * 100)

        return jsonify({
            "digit": digit,
            "confidence": round(confidence, 2)
        })

    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 400


if __name__ == "__main__":
    load_trained_model()
    app.run(host="127.0.0.1", port=5000, debug=True)
