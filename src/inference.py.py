# src/inference.py

import cv2
import numpy as np
import onnxruntime as ort
import argparse
import json
import os

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "weights/visual_model.onnx"
TEXT_FEATURES_PATH = "weights/text_features.npy"
LABELS_PATH = "weights/labels.json"   # optional (recommended)

# -------------------------------
# LOAD MODEL
# -------------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    return session

def load_text_features():
    if not os.path.exists(TEXT_FEATURES_PATH):
        raise FileNotFoundError(f"text_features.npy not found at {TEXT_FEATURES_PATH}")

    return np.load(TEXT_FEATURES_PATH)

def load_labels():
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r") as f:
            return json.load(f)
    else:
        # fallback (basic categories)
        return ["plastic", "paper", "glass", "metal", "organic", "e-waste"]

# -------------------------------
# PREPROCESS
# -------------------------------
def preprocess(image):
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = (image - [0.481, 0.457, 0.408]) / [0.268, 0.261, 0.275]
    image = np.transpose(image, (2, 0, 1))
    return np.expand_dims(image, axis=0).astype(np.float32)

# -------------------------------
# PREDICTION
# -------------------------------
def predict(image_path, session, text_features, labels):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Invalid image format")

    image = preprocess(image)

    # Run model
    image_features = session.run(
        ["image_features"],
        {"pixel_values": image}
    )[0]

    # Normalize
    image_features /= np.linalg.norm(image_features, axis=1, keepdims=True)

    # Similarity
    logits = np.dot(image_features, text_features.T)

    # Softmax
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    idx = int(np.argmax(probs))
    confidence = float(probs[0][idx])

    label = labels[idx] if idx < len(labels) else f"class_{idx}"

    return label, confidence

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Waste Classification Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")

    args = parser.parse_args()

    print("Loading model...")
    session = load_model()
    text_features = load_text_features()
    labels = load_labels()

    print("Running prediction...")
    label, confidence = predict(args.image, session, text_features, labels)

    print("\n--- RESULT ---")
    print(f"Prediction : {label}")
    print(f"Confidence : {confidence:.2f}")
