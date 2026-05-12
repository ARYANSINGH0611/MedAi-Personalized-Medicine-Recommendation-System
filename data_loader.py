"""
data_loader.py  –  Load, clean, and index all supporting datasets.

All expensive I/O happens once at import time (or on explicit call).
Every lookup is O(1) via pre-built dicts.
"""

import pandas as pd
from utils import clean_list_column


# ──────────────────────────────────────────────
#  Internal loaders
# ──────────────────────────────────────────────

def _load_description(path: str) -> dict[str, str]:
    df = pd.read_csv(path, encoding="latin1", on_bad_lines="skip")
    df["Disease"] = df["Disease"].str.strip()
    return dict(zip(df["Disease"], df["Description"]))


def _load_precautions(path: str) -> dict[str, list[str]]:
    df = pd.read_csv(path, encoding="latin1", on_bad_lines="skip")
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df["Disease"] = df["Disease"].str.strip()
    result = {}
    for _, row in df.iterrows():
        disease = row["Disease"]
        prec = [
            str(v).strip()
            for v in row.iloc[1:]
            if pd.notna(v) and str(v).strip() not in ("", "nan", disease)
        ]
        result[disease] = prec
    return result


def _load_medications(path: str) -> dict[str, list[str]]:
    df = pd.read_csv(path, encoding="latin1", on_bad_lines="skip")
    df["Disease"] = df["Disease"].str.strip()
    df["Medication"] = df["Medication"].apply(clean_list_column)
    return dict(zip(df["Disease"], df["Medication"]))


def _load_diets(path: str) -> dict[str, list[str]]:
    df = pd.read_csv(path, encoding="latin1", on_bad_lines="skip")
    df["Disease"] = df["Disease"].str.strip()
    df["Diet"] = df["Diet"].apply(clean_list_column)
    return dict(zip(df["Disease"], df["Diet"]))


def _load_workout(path: str) -> dict[str, list[str]]:
    df = pd.read_csv(path, encoding="latin1", on_bad_lines="skip")
    df = df.rename(columns={"disease": "Disease"})
    df["Disease"] = df["Disease"].str.strip()
    result = {}
    for disease, grp in df.groupby("Disease"):
        result[disease] = grp["workout"].dropna().tolist()
    return result

def _load_doctors(path: str) -> dict[str, dict]:
    # ── Column name constants (update here if CSV headers change) ──
    COL_DISEASE  = "Disease"
    COL_CATEGORY = "Category"
    COL_DELHI    = "Top Doctors (Delhi)"
    COL_UK       = "Top Doctors (Uttarakhand)"

    df = pd.read_csv(path, encoding="latin1", on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]
    df[COL_DISEASE] = df[COL_DISEASE].str.strip()

    result = {}
    for _, row in df.iterrows():
        disease = row[COL_DISEASE]
        result[disease] = {
            "category": str(row.get(COL_CATEGORY, "")).strip(),
            "delhi": [
                d.strip()
                for d in str(row.get(COL_DELHI, "")).split(",")
                if d.strip()
            ],
            "uttarakhand": [
                d.strip()
                for d in str(row.get(COL_UK, "")).split(",")
                if d.strip()
            ],
        }
    return result

def _load_severity(path: str) -> dict[str, int]:
    df = pd.read_csv(path)
    df["Symptom"] = df["Symptom"].str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    return dict(zip(df["Symptom"], df["weight"]))


# ──────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────

DATASET_DIR = "datasets"

def load_all(dataset_dir: str = DATASET_DIR) -> dict:
    """
    Load every supporting dataset and return a single dict of fast lookups.

    Keys
    ----
    descriptions  : {disease: str}
    precautions   : {disease: [str]}
    medications   : {disease: [str]}
    diets         : {disease: [str]}
    workouts      : {disease: [str]}
    severity      : {symptom: int}

    """
    d = dataset_dir.rstrip("/")
    return {
        "descriptions": _load_description(f"{d}/description.csv"),
        "precautions":  _load_precautions(f"{d}/precautions_df.csv"),
        "medications":  _load_medications(f"{d}/medications.csv"),
        "diets":        _load_diets(f"{d}/diets.csv"),
        "workouts":     _load_workout(f"{d}/workout_df.csv"),
        "severity":     _load_severity(f"{d}/Symptom-severity.csv"),
        "doctors": _load_doctors(f"{d}/doctors.csv"),
    }


def get_details(disease: str, lookups: dict) -> dict:
    """
    Fetch all recommendation details for a predicted disease.

    Returns a dict with keys: description, precautions, medications,
    workouts, diets  (all guaranteed to be strings or lists).
    """
    doctor_data = lookups["doctors"].get(disease, {})

    return {
    "description": lookups["descriptions"].get(disease, "Not available."),
    "precautions": lookups["precautions"].get(disease, []),
    "medications": lookups["medications"].get(disease, []),
    "workouts":    lookups["workouts"].get(disease, []),
    "diets":       lookups["diets"].get(disease, []),
    "category": doctor_data.get("category", "Not available"),
    "doctors_delhi": doctor_data.get("delhi", []),
    "doctors_uk": doctor_data.get("uttarakhand", []),
}
