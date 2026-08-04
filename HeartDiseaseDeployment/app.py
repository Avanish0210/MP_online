"""Flask API for serving the heart disease prediction model."""
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

app = Flask(__name__)
model = joblib.load(MODEL_PATH)


def validate_payload(payload):
    """Return a numeric one-row DataFrame or a helpful validation error."""
    if not isinstance(payload, dict):
        return None, "Request body must be a JSON object."
    missing = [feature for feature in FEATURES if feature not in payload]
    if missing:
        return None, f"Missing required fields: {', '.join(missing)}."
    try:
        values = {feature: float(payload[feature]) for feature in FEATURES}
    except (TypeError, ValueError):
        return None, "Every feature value must be numeric."
    return pd.DataFrame([values], columns=FEATURES), None


@app.get("/")
def home():
    return render_template("index.html", features=FEATURES)


@app.get("/health")
def health():
    return jsonify(status="healthy", model_loaded=True)


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    input_frame, error = validate_payload(payload)
    if error:
        return jsonify(error=error, required_features=FEATURES), 400

    prediction = int(model.predict(input_frame)[0])
    probability = float(model.predict_proba(input_frame)[0][1])
    label = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"
    return jsonify(
        prediction=label,
        prediction_code=prediction,
        heart_disease_probability=round(probability, 4),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
