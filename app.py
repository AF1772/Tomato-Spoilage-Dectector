"""
app.py
──────
Flask 3.1.3 web application for real-time tomato spoilage detection.

Changes from Flask 3.0 → 3.1.3:
  - werkzeug.utils.secure_filename is still the correct import path
  - Response typing is more strict — jsonify remains unchanged
  - Error handlers use @app.errorhandler decorator (unchanged)

Changes from TF 2.13 → TF 2.20:
  - keras.models.load_model replaces tf.keras.models.load_model
  - Preprocessing (Rescaling) is applied as a Keras layer inside the pipeline
  - Model.predict returns numpy array directly

Run with:
    python app.py
Open:   http://localhost:5000
"""

import os
import io
import base64
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image

# ── Configuration ──────────────────────────────────────────────────────────────
ARCH               = "mobilenetv2"
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", f"{ARCH}_final.keras")
IMG_SIZE           = (224, 224)
THRESHOLD          = 0.25      # P(Spoiled) >= 0.25 → Spoiled
CAUTION_THRESHOLD  = 0.7      # flag for human review if confidence < 70%
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTS       = {"jpg", "jpeg", "png", "webp"}
MAX_UPLOAD_MB      = 16
# ───────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Load model at startup ──────────────────────────────────────────────────────
# Model is loaded once when the app starts, not on every request.
# This keeps inference fast (under 2 seconds per image).
print(f"\n{'─' * 48}")
print(f"  Loading model: {MODEL_PATH}")
try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"  ✓ Model loaded: {model.name}")
    print(f"  ✓ Input shape : {model.input_shape}")
except Exception as exc:
    model = None
    print(f"  ✗ Model not loaded: {exc}")
    print(f"  → Run: python src/train.py  then restart the app.")
print(f"{'─' * 48}\n")



# ── Utility functions ──────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS
    )


def preprocess_image(img_bytes: bytes):
    """
    Preprocesses an uploaded image to match the training pipeline (Section 3.5):
      1. Open as RGB (handles PNG transparency, EXIF rotation)
      2. Resize to 224×224 using bilinear interpolation
      3. Convert to float32 array
      4. Normalise pixels to [0.0, 1.0]
      5. Add batch dimension → shape (1, 224, 224, 3)

    Returns:
        img_batch  : numpy array ready for model.predict()
        img_thumb  : PIL image for display thumbnail
    """
    img         = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_resized = img.resize(IMG_SIZE, Image.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32)  # keep as [0, 255]
    img_batch   = np.expand_dims(img_array, axis=0)
    return img_batch, img_resized


def image_to_base64(pil_img: Image.Image, quality: int = 85) -> str:
    """Encodes a PIL image to a base64 JPEG string for JSON embedding."""
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def classify(prob_spoiled: float) -> dict:
    """
    Applies decision threshold and caution flagging (Section 3.9).

    Decision rule:
        P(Spoiled) >= 0.25  → Spoiled
        P(Spoiled) <  0.25  → Fresh

    Caution flag:
        If the confidence on the predicted class is below 70%, the
        result is flagged for human review. This retains human oversight
        for borderline cases — consistent with the explainability
        requirements discussed in Chapter 2.
    """
    label      = "Spoiled" if prob_spoiled >= THRESHOLD else "Fresh"
    prob_fresh = 1.0 - prob_spoiled
    confidence = prob_spoiled if label == "Spoiled" else prob_fresh

    needs_review = confidence < CAUTION_THRESHOLD
    review_note  = (
        "⚠️ Confidence below 70% — recommend human review before acting on this result"
        if needs_review else ""
    )

    return {
        "label":        label,
        "confidence":   round(float(confidence) * 100, 2),
        "prob_fresh":   round(float(prob_fresh)  * 100, 2),
        "prob_spoiled": round(float(prob_spoiled) * 100, 2),
        "needs_review": bool(needs_review),
        "review_note":  review_note
    }


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the main detection interface."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST endpoint — accepts multipart/form-data with field 'file'.
    Returns JSON with classification label, confidence, probabilities,
    caution flag, and a base64 thumbnail of the uploaded image.
    """
    if model is None:
        return jsonify({
            "error": "Model not loaded. Run: python src/train.py first, then restart app.py."
        }), 503

    if "file" not in request.files:
        return jsonify({"error": "No file field in the request."}), 400

    file = request.files["file"]

    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"File type not supported. Use: {', '.join(ALLOWED_EXTS)}"
        }), 415

    try:
        img_bytes = file.read()

        if len(img_bytes) == 0:
            return jsonify({"error": "Uploaded file is empty."}), 400

        img_batch, img_thumb = preprocess_image(img_bytes)

        # Run model inference
        raw_output   = model.predict(img_batch, verbose=0)
        prob_spoiled = float(raw_output[0][0])

        result              = classify(prob_spoiled)
        result["thumbnail"] = image_to_base64(img_thumb)

        return jsonify(result), 200

    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {str(exc)}"}), 500


@app.route("/health")
def health():
    """
    Health check endpoint.
    Useful for verifying the server is up and the model is loaded.
    GET http://localhost:5000/health
    """
    return jsonify({
        "status":       "ok",
        "model_loaded": model is not None,
        "architecture": ARCH,
        "tf_version":   tf.__version__,
        "threshold":    THRESHOLD
    }), 200


# ── Error handlers ─────────────────────────────────────────────────────────────
@app.errorhandler(413)
def too_large(e):
    return jsonify({
        "error": f"File too large. Maximum size is {MAX_UPLOAD_MB}MB."
    }), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting Flask development server...")
    print("Open http://localhost:5000 in your browser\n")
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000,
        use_reloader=False    # set False to avoid loading the model twice
    )