# ══════════════════════════════════════════════════════════════
#  NeuroScan AI — Flask Backend
#  Brain Tumor MRI Classification
#  Run:  python app.py
#  Open: http://127.0.0.1:5000
# ══════════════════════════════════════════════════════════════

import os
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model


# ── App setup ─────────────────────────────────────────────────
app = Flask(__name__, static_folder='static', template_folder='.')

UPLOAD_FOLDER    = 'uploads'
ALLOWED_EXT      = {'jpg', 'jpeg', 'png'}
MODEL_PATH       = 'model.h5'
IMG_SIZE         = (150, 150)

# Matches the order your model was trained with (flow_from_directory alphabetical)
CLASS_NAMES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
CLASS_KEYS  = ['glioma',       'meningioma',        'noTumor',  'pituitary']

app.config['UPLOAD_FOLDER']      = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024   # 10 MB limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Load model once at startup ────────────────────────────────
model = None
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print(f"✅  Model loaded from '{MODEL_PATH}'")
    model.summary()
else:
    print(f"⚠️   '{MODEL_PATH}' not found.")
    print("     Run  python train.py  first to train and save the model.")


# ── Helpers ───────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT
    )


def preprocess(filepath: str) -> np.ndarray:
    """
    Load image → resize to 150×150 → normalize to [0,1]
    → reshape to (1, 150, 150, 3) for model.predict()
    """
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError("Could not read the image file. Make sure it is a valid JPG/PNG.")
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype('float32') / 255.0
    img = np.reshape(img, (1, IMG_SIZE[0], IMG_SIZE[1], 3))
    return img


# ── Routes ────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the frontend HTML file."""
    return send_from_directory('.', 'app.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts a multipart/form-data POST with key 'file'.
    Returns JSON:
    {
        "class":       "Glioma Tumor",
        "confidence":  87.43,
        "probabilities": {
            "glioma":     87.43,
            "meningioma":  5.21,
            "noTumor":     4.10,
            "pituitary":   3.26
        }
    }
    """

    # ── Guard: model not loaded ──
    if model is None:
        return jsonify({
            'error': "Model not found. Please run 'python train.py' first."
        }), 500

    # ── Guard: file in request ──
    if 'file' not in request.files:
        return jsonify({'error': 'No file sent. Include an image with key "file".'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only JPG and PNG are accepted.'}), 400

    # ── Save temporarily ──
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # ── Preprocess ──
        img = preprocess(filepath)

        # ── Inference ──
        prediction = model.predict(img, verbose=0)   # shape: (1, 4)
        probs      = prediction[0].tolist()           # list of 4 floats

        predicted_idx   = int(np.argmax(probs))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence      = round(float(np.max(probs)) * 100, 2)

        # Build probabilities dict
        prob_dict = {
            CLASS_KEYS[i]: round(probs[i] * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        response = {
            'class':         predicted_class,
            'confidence':    confidence,
            'probabilities': prob_dict
        }

        print(f"📊  Prediction: {predicted_class}  ({confidence}%)")
        return jsonify(response)

    except Exception as e:
        print(f"❌  Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

    finally:
        # Always clean up the uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)


# ── Health check ──────────────────────────────────────────────
@app.route('/health')
def health():
    return jsonify({
        'status':       'ok',
        'model_loaded': model is not None,
        'classes':      CLASS_NAMES
    })


# ── Run ───────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "═" * 50)
    print("  🧠  NeuroScan AI — Brain Tumor Detection")
    print("═" * 50)
    print(f"  Model : {MODEL_PATH}")
    print(f"  URL   : http://127.0.0.1:5001")
    print("═" * 50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5001)