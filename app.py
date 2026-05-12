"""
app.py  –  Flask web interface for the Medicine Recommendation System

Uses the modular pipeline from train.py:
  - RF model for >= 4 symptoms
  - SVM model for < 4 symptoms
  - All lookups via data_loader (O(1) dicts)
  - Fuzzy symptom matching via utils

Run:
    python app.py
"""

from flask import Flask, render_template, request, jsonify
import pickle
import re

from data_loader import load_all, get_details
from utils import (
    build_normalised_map,
    parse_user_input,
    symptoms_to_vector,
    calibrate_confidence,
)

app = Flask(__name__)

# ── Load once at startup ───────────────────────────────────────

with open("medical_model.pkl", "rb") as f:
    pipeline = pickle.load(f)

rf_model     = pipeline["rf_model"]
svm_model    = pipeline["svm_model"]
le           = pipeline["label_encoder"]
all_symptoms = pipeline["all_symptoms"]
importances  = pipeline.get("importances", {})
metadata     = pipeline.get("metadata", {})
THRESHOLD    = pipeline.get("threshold", 4)

lookups  = load_all("datasets")
norm_map = build_normalised_map(all_symptoms)

DANGEROUS = {"AIDS", "Paralysis (brain hemorrhage)", "Heart attack"}


# ── Prediction logic ──────────────────────────────────────────

def run_prediction(user_input: str) -> dict:
    user_symptoms, unknown = parse_user_input(user_input, norm_map)
    n = len(user_symptoms)

    if not user_symptoms:
        return {"error": "No valid symptoms recognised. Please try again."}

    vec   = symptoms_to_vector(user_symptoms, all_symptoms)
    model = rf_model if n >= THRESHOLD else svm_model
    model_name = "Random Forest" if n >= THRESHOLD else "SVM (linear)"

    pred_enc   = model.predict(vec)[0]
    disease    = le.inverse_transform([pred_enc])[0]
    probs      = model.predict_proba(vec)[0]
    sorted_probs = sorted(probs, reverse=True)
    prob_gap   = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
    confidence, conf_label = calibrate_confidence(probs.max(), n, user_symptoms, lookups["severity"], prob_gap)

    # Severity score
    sev = lookups["severity"]
    severity_score = sum(sev.get(re.sub(r"_+", "_", s.lower().replace(" ", "_")), 0) for s in user_symptoms)
    max_possible   = n * 7
    sev_pct        = severity_score / max_possible if max_possible else 0
    severity_label = "Critical" if sev_pct >= 0.75 else "High" if sev_pct >= 0.5 else "Moderate" if sev_pct >= 0.25 else "Low"

    # Top-3 alternatives
    top3 = probs.argsort()[-3:][::-1]
    alternatives = [
        {"disease": le.inverse_transform([i])[0], "prob": round(float(probs[i]) * 100, 1)}
        for i in top3
        if probs[i] * 100 >= 5
    ]

    # Feature contributions (RF only)
    contributions = []
    if model_name == "Random Forest":
        contributions = sorted(
            [{"symptom": s.replace("_", " "), "score": round(importances.get(s, 0), 5)}
             for s in user_symptoms],
            key=lambda x: x["score"], reverse=True
        )[:5]

    low_confidence = confidence < 50 or n < 3

    if low_confidence:
        return {
            "mode":           "low",
            "confidence":     confidence,
            "conf_label":     conf_label,
            "n_symptoms":     n,
            "model_used":     model_name,
            "severity_score": severity_score,
            "severity_label": severity_label,
            "alternatives": [
                a for a in alternatives
                if not (a["disease"] in DANGEROUS and a["prob"] < 60)
            ],
            "unknown": unknown,
        }

    details = get_details(disease, lookups)
    return {
    "mode":           "high",
    "disease":        disease,
    "confidence":     confidence,
    "conf_label":     conf_label,
    "n_symptoms":     n,
    "model_used":     model_name,
    "severity_score": severity_score,
    "severity_label": severity_label,
    "alternatives":   alternatives,
    "contributions":  contributions,
    "description":    details["description"],
    "precautions":    details["precautions"],
    "medications":    details["medications"],
    "workouts":       list(details["workouts"]),
    "diets":          details["diets"],
    "category":       details["category"],
    "doctors_delhi":  details["doctors_delhi"],
    "doctors_uk":     details["doctors_uk"],
    "unknown":        unknown,
}


# ── Routes ────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        all_symptoms=[s.replace("_", " ") for s in all_symptoms],
        metadata=metadata,
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    user_input = data.get("symptoms", "").strip()
    if not user_input:
        return jsonify({"error": "No symptoms provided."}), 400
    return jsonify(run_prediction(user_input))


@app.route("/details/<path:disease>", methods=["GET"])
def disease_details(disease: str):
    """Return full details for any disease by name."""
    details = get_details(disease, lookups)
    return jsonify({
        "disease":     disease,
        "description": details["description"],
        "precautions": details["precautions"],
        "medications": details["medications"],
        "workouts":    list(details["workouts"]),
        "diets":       details["diets"],
        "category":      details["category"],
"doctors_delhi": details["doctors_delhi"],
"doctors_uk":    details["doctors_uk"],
    })


@app.route("/symptoms", methods=["GET"])
def symptoms_list():
    """Return all valid symptoms for autocomplete."""
    return jsonify([s.replace("_", " ") for s in all_symptoms])


if __name__ == "__main__":
    app.run(debug=False)