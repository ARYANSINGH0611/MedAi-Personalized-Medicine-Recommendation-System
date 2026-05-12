"""
predict.py  –  Interactive Medicine Recommendation CLI

Loads the trained pipeline from medical_model.pkl and provides
a clean terminal interface for symptom input and disease prediction.

Model selection strategy
------------------------
  ≥ 4 symptoms  →  Random Forest  (better at full context, gives feature importance)
  < 4 symptoms  →  SVM linear     (outperforms RF on sparse / few-symptom input)

Usage
-----
  python predict.py
"""

import pickle
import sys
import textwrap
from pathlib import Path

from data_loader import load_all, get_details
from utils import (
    build_normalised_map,
    parse_user_input,
    symptoms_to_vector,
    calibrate_confidence,
)

MODEL_PATH   = "medical_model.pkl"
DATASET_DIR  = "datasets"

# ── ANSI palette ──────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BLUE   = "\033[94m"
MAGENTA= "\033[95m"
WHITE  = "\033[97m"

# Diseases that need a higher confidence threshold before showing
DANGEROUS_DISEASES = {"AIDS", "Paralysis (brain hemorrhage)", "Heart attack"}


# ──────────────────────────────────────────────
#  Terminal drawing helpers
# ──────────────────────────────────────────────

def clear_line() -> None:
    print("\r" + " " * 70 + "\r", end="", flush=True)

def divider(char: str = "─", width: int = 58, color: str = DIM) -> None:
    print(f"  {color}{char * width}{RESET}")

def header(title: str) -> None:
    print(f"\n  {BOLD}{CYAN}{title}{RESET}")
    divider("─", 58, CYAN)

def subheader(icon: str, title: str) -> None:
    print(f"\n  {BOLD}{icon}  {title}{RESET}")
    divider()

def bullet(text: str, idx: int | None = None, color: str = WHITE) -> None:
    prefix = f"{DIM}{idx}.{RESET}" if idx is not None else f"{DIM}•{RESET}"
    wrapped = textwrap.fill(text, width=70, subsequent_indent="       ")
    print(f"    {prefix}  {color}{wrapped}{RESET}")

def badge(label: str, value: str, color: str = CYAN) -> None:
    print(f"  {DIM}{label:<22}{RESET}{color}{BOLD}{value}{RESET}")

def confidence_bar(pct: float, width: int = 30) -> str:
    filled = int(pct / 100 * width)
    bar    = "█" * filled + "░" * (width - filled)
    if pct >= 80:
        color = GREEN
    elif pct >= 50:
        color = YELLOW
    else:
        color = RED
    return f"{color}{bar}{RESET}  {BOLD}{pct:.1f}%{RESET}"


# ──────────────────────────────────────────────
#  Load pipeline
# ──────────────────────────────────────────────

def load_pipeline(path: str) -> dict:
    if not Path(path).exists():
        print(f"\n  {RED}✘  Model file not found: {path}{RESET}")
        print(f"     Run  python train.py  first.\n")
        sys.exit(1)

    with open(path, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────
#  Prediction
# ──────────────────────────────────────────────

def predict(pipeline: dict, user_symptoms: list[str], severity: dict | None = None) -> dict:
    """
    Choose RF or SVM based on symptom count and return full prediction info.
    """
    n   = len(user_symptoms)
    rf  = pipeline["rf_model"]
    svm = pipeline["svm_model"]
    le  = pipeline["label_encoder"]
    all_symptoms = pipeline["all_symptoms"]

    vec = symptoms_to_vector(user_symptoms, all_symptoms)

    # Model selection — threshold auto-detected during training
    threshold  = pipeline.get("threshold", 4)
    if n >= threshold:
        model      = rf
        model_name = "Random Forest"
    else:
        model      = svm
        model_name = "SVM (linear)"

    predicted_enc = model.predict(vec)[0]
    disease       = le.inverse_transform([predicted_enc])[0]
    probs         = model.predict_proba(vec)[0]
    raw_prob      = probs.max()
    sorted_probs  = sorted(probs, reverse=True)
    prob_gap      = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0

    confidence, conf_label = calibrate_confidence(raw_prob, n, user_symptoms, severity, prob_gap)

    # Top-3 alternatives
    top3_idx = probs.argsort()[-3:][::-1]
    alternatives = [
        (le.inverse_transform([i])[0], round(probs[i] * 100, 2))
        for i in top3_idx
        if probs[i] * 100 >= 5
    ]

    # Feature contributions (RF only)
    contributions = []
    if model_name == "Random Forest" and "importances" in pipeline:
        importances = pipeline["importances"]
        contributions = sorted(
            [(s, importances.get(s, 0)) for s in user_symptoms],
            key=lambda x: x[1],
            reverse=True,
        )[:5]

    return {
        "disease":       disease,
        "model_used":    model_name,
        "confidence":    confidence,
        "conf_label":    conf_label,
        "n_symptoms":    n,
        "alternatives":  alternatives,
        "contributions": contributions,
    }


# ──────────────────────────────────────────────
#  Report renderer
# ──────────────────────────────────────────────

def print_report(pred: dict, details: dict) -> None:
    print(f"\n\n  {BOLD}{CYAN}{'═' * 58}{RESET}")
    print(f"  {BOLD}{CYAN}  PERSONALISED MEDICAL REPORT{RESET}")
    print(f"  {BOLD}{CYAN}{'═' * 58}{RESET}\n")

    low_confidence = pred["confidence"] < 50 or pred["n_symptoms"] < 3

    # ── Prediction block ──
    if low_confidence:
        print(f"  {YELLOW}{BOLD}⚠  Low confidence — possible conditions:{RESET}\n")
        for disease, prob in pred["alternatives"]:
            if disease in DANGEROUS_DISEASES and prob < 60:
                continue
            bar_w = int(prob / 100 * 20)
            bar   = f"{YELLOW}{'█' * bar_w}{'░' * (20 - bar_w)}{RESET}"
            print(f"    {bar}  {prob:5.1f}%  {disease}")
        print()
    else:
        badge("Predicted disease  :", pred["disease"], GREEN)
        badge("Model used         :", pred["model_used"], BLUE)

    # Confidence bar
    print(f"\n  {DIM}{'Confidence':<22}{RESET}{confidence_bar(pred['confidence'])}")
    badge("Confidence label   :", pred["conf_label"],
          GREEN if pred["conf_label"] == "High" else YELLOW if pred["conf_label"] == "Moderate" else RED)
    badge("Symptoms entered   :", str(pred["n_symptoms"]))

    if low_confidence:
        print(f"\n  {YELLOW}ℹ  Provide more symptoms for a reliable prediction.{RESET}")
        divider("═", 58, CYAN)
        return

    # ── Feature contributions ──
    if pred["contributions"]:
        subheader("◈", "Key symptoms driving this prediction")
        max_imp = pred["contributions"][0][1] if pred["contributions"] else 1
        for sym, imp in pred["contributions"]:
            bar_w = int(imp / max_imp * 20) if max_imp > 0 else 0
            bar   = f"{BLUE}{'▪' * bar_w}{'·' * (20 - bar_w)}{RESET}"
            print(f"    {bar}  {sym.replace('_', ' ')}")

    # ── Alternatives ──
    if len(pred["alternatives"]) > 1:
        subheader("◉", "Alternative possibilities")
        for i, (d, prob) in enumerate(pred["alternatives"][1:], 1):
            print(f"    {DIM}{i}.{RESET}  {d}  {DIM}({prob:.1f}%){RESET}")

    # ── Description ──
    subheader("📋", "About this condition")
    desc = details.get("description", "Not available.")
    wrapped = textwrap.fill(desc, width=70, initial_indent="    ", subsequent_indent="    ")
    print(f"{DIM}{wrapped}{RESET}")

    # ── Precautions ──
    pre = details.get("precautions", [])
    if pre:
        subheader("⚠", "Precautions")
        for i, p in enumerate(pre, 1):
            bullet(p, idx=i, color=YELLOW)

    # ── Medications ──
    med = details.get("medications", [])
    if med:
        subheader("💊", "Commonly associated medications")
        print(f"  {DIM}(These are for informational purposes only — consult a doctor){RESET}\n")
        for i, m in enumerate(med, 1):
            bullet(m, idx=i, color=WHITE)

    # ── Workout ──
    work = details.get("workouts", [])
    if work:
        subheader("🏃", "Recommended activity / lifestyle")
        for i, w in enumerate(work, 1):
            bullet(w, idx=i, color=GREEN)

    # ── Diet ──
    diet = details.get("diets", [])
    if diet:
        subheader("🥗", "Diet recommendations")
        for i, d in enumerate(diet, 1):
            bullet(d, idx=i, color=MAGENTA)

    # ── Disclaimer ──
    print(f"\n  {CYAN}{'─' * 58}{RESET}")
    print(f"  {YELLOW}{BOLD}⚠  DISCLAIMER{RESET}")
    print(f"  {DIM}This is an AI-based recommendation tool and does NOT replace{RESET}")
    print(f"  {DIM}professional medical advice. Always consult a qualified doctor.{RESET}")
    print(f"  {CYAN}{'═' * 58}{RESET}\n")


# ──────────────────────────────────────────────
#  Main loop
# ──────────────────────────────────────────────

def main():
    print(f"\n  {BOLD}{CYAN}{'═' * 58}{RESET}")
    print(f"  {BOLD}{CYAN}  Medicine Recommendation System{RESET}")
    print(f"  {CYAN}  Loading models …{RESET}")

    pipeline     = load_pipeline(MODEL_PATH)
    lookups      = load_all(DATASET_DIR)
    all_symptoms = pipeline["all_symptoms"]
    norm_map     = build_normalised_map(all_symptoms)

    meta = pipeline.get("metadata", {})
    print(f"  {GREEN}✔  Pipeline loaded{RESET}  "
          f"{DIM}(RF acc {meta.get('rf_cv_acc', '?'):.4f} · "
          f"trained {str(meta.get('trained_on', ''))[:10]}){RESET}")
    print(f"  {CYAN}{'═' * 58}{RESET}\n")
    print(f"  {DIM}Type symptoms separated by commas, or 'quit' to exit.{RESET}")
    print(f"  {DIM}Example: fever, nausea, headache, fatigue{RESET}\n")

    while True:
        try:
            user_input = input(f"  {BOLD}{CYAN}Symptoms ›{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {DIM}Goodbye.{RESET}\n")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print(f"\n  {DIM}Goodbye.{RESET}\n")
            break

        user_symptoms, unknown = parse_user_input(user_input, norm_map)

        if unknown:
            print(f"\n  {YELLOW}Unrecognised:{RESET} {', '.join(unknown)}")

        if not user_symptoms:
            print(f"  {RED}✘  No valid symptoms found. Try again.{RESET}\n")
            continue

        pred    = predict(pipeline, user_symptoms, lookups["severity"])
        details = get_details(pred["disease"], lookups)
        print_report(pred, details)


if __name__ == "__main__":
    main()
