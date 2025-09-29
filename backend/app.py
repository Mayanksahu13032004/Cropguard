import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# --------------------------
# Load Model & Labels
# --------------------------
MODEL_PATH = "model.h5"
LABELS_PATH = "labels.txt"

model = load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# --------------------------
# Disease → Solution Mapping
# --------------------------
solutions = {
    "Apple___Scab": "Spray fungicides like Mancozeb or Captan. Remove infected leaves.",
    "Apple___Black_rot": "Prune infected branches. Apply fungicides (e.g., Thiophanate-methyl).",
    "Corn___Rust": "Use resistant hybrids. Apply fungicides at early stages.",
    "Tomato___Late_blight": "Spray copper-based fungicides. Avoid overhead irrigation.",
    "Healthy": "Your crop is healthy! Keep monitoring and maintain good care."
}

# --------------------------
# Flask App
# --------------------------
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    img_path = os.path.join("temp.jpg")
    file.save(img_path)

    # Preprocess image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    class_index = np.argmax(preds)
    predicted_class = labels[class_index]
    confidence = float(np.max(preds))

    # Solution
    solution = solutions.get(predicted_class, "No solution available.")

    return jsonify({
        "prediction": predicted_class,
        "confidence": confidence,
        "solution": solution
    })

if __name__ == "__main__":
    app.run(debug=True)
