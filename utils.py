"""
utils.py  –  Shared helpers for the Medicine Recommendation System
"""

import ast
import re
from difflib import get_close_matches


# ──────────────────────────────────────────────
#  List / string cleaning
# ──────────────────────────────────────────────

def clean_list_column(val):
    """Parse a stringified Python list into a real list."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            val = val.strip("[]")
            return [v.strip().strip("'\"") for v in val.split(",") if v.strip()]
    return []


# ──────────────────────────────────────────────
#  Symptom normalisation
# ──────────────────────────────────────────────

def build_normalised_map(all_symptoms: list[str]) -> dict[str, str]:
    """
    Build a lookup: normalised human string  →  raw column name.
    E.g. 'high fever' → 'high_fever'
    """
    mapping = {}
    for sym in all_symptoms:
        key = sym.replace("_", " ").strip().lower()
        mapping[key] = sym
    return mapping


def parse_user_input(user_input: str, normalised_map: dict) -> tuple[list[str], list[str]]:
    """
    Convert a comma-separated symptom string into validated column names.

    Returns
    -------
    parsed   : list of matched column names
    unknown  : list of tokens that could not be matched
    """
    parsed, unknown = [], []

    for token in user_input.split(","):
        token = token.strip().lower().replace("_", " ")
        token = re.sub(r"\s+", " ", token)          # collapse whitespace

        if token in normalised_map:
            parsed.append(normalised_map[token])
            continue

        # fuzzy match
        close = get_close_matches(token, normalised_map.keys(), n=1, cutoff=0.80)
        if close:
            matched_col = normalised_map[close[0]]
            print(f"  ⟳  '{token}' interpreted as '{close[0]}'")
            parsed.append(matched_col)
        else:
            unknown.append(token)

    return parsed, unknown


# ──────────────────────────────────────────────
#  Feature vector construction
# ──────────────────────────────────────────────

def symptoms_to_vector(user_symptoms: list[str], all_symptoms: list[str]):
    """
    Build a binary feature vector (DataFrame row) from symptom column names.
    Uses BINARY encoding to match training data (0/1 only).
    """
    import pandas as pd
    vector = {s: 0 for s in all_symptoms}
    for s in user_symptoms:
        if s in vector:
            vector[s] = 1
    return pd.DataFrame([vector])


# ──────────────────────────────────────────────
#  Confidence calibration
# ──────────────────────────────────────────────

def calibrate_confidence(
    raw_prob: float,
    n_symptoms: int,
    user_symptoms: list[str] | None = None,
    severity: dict[str, int] | None = None,
    prob_gap: float | None = None,
) -> tuple[float, str]:
    """
    Heuristic confidence calibration.

    Adjustments applied (in order):
      1. Sparse-input penalty  — if < 3 symptoms, scale down by 40 %
      2. Prob-gap adjustment   — gap between top-1 and top-2 prob:
                                  large gap → boost, small gap → penalty (±10 pp)
      3. Severity adjustment   — avg symptom severity vs midpoint 4 (±5 pp)
      4. Cap extremes above 95 %

    Returns calibrated confidence (0–100) and a label string.
    """
    confidence = raw_prob * 100

    # 1. Sparse penalty
    if n_symptoms < 3:
        confidence *= 0.60

    # 2. Probability gap: gap=1.0 means certain, gap≈0 means a toss-up
    if prob_gap is not None:
        # centre around 0.5 → adjustment in [-10, +10] pp
        gap_factor = (prob_gap - 0.5) * 20
        confidence += gap_factor

    # 3. Severity adjustment
    if user_symptoms and severity:
        scores = [severity.get(re.sub(r"_+", "_", s.lower().replace(" ", "_")), 4) for s in user_symptoms]
        avg_sev = sum(scores) / len(scores)
        sev_factor = (avg_sev - 4) / 3              # maps to [-1, +1]
        confidence += sev_factor * 5                 # ±5 pp

    # 4. Cap extremes
    if confidence > 95:
        confidence = 95 + (confidence - 95) * 0.20

    confidence = min(max(confidence, 0.0), 100.0)

    if confidence < 50:
        label = "Low"
    elif confidence < 80:
        label = "Moderate"
    else:
        label = "High"

    return round(confidence, 2), label
