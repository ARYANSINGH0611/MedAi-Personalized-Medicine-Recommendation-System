"""
train.py  –  Training pipeline for the Medicine Recommendation System

Architecture
------------
  Primary model  : RandomForestClassifier
                   • Best overall accuracy + explainability (feature importance)
                   • Robust to the 94 % sparse binary feature space
                   • Fast training (~2 s) and instant inference

  Backup model   : SVC (linear kernel)
                   • Outperforms RF when user enters ≤ 3 symptoms
                   • Handles high-dimensional sparse inputs natively

  Selection rule : RF is used when ≥ 4 symptoms are entered.
                   SVM is used for sparse input (< 4 symptoms).

Usage
-----
  python train.py
"""

import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

DATASET_PATH = "datasets/Training.csv"
OUTPUT_PATH  = "medical_model.pkl"

# ── ANSI colours (terminal UI) ────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"
BLUE   = "\033[94m"

def banner(text: str, char: str = "─", width: int = 60) -> None:
    line = char * width
    print(f"\n{CYAN}{line}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{CYAN}{line}{RESET}")

def success(msg: str) -> None:
    print(f"  {GREEN}✔{RESET}  {msg}")

def info(msg: str) -> None:
    print(f"  {BLUE}◈{RESET}  {msg}")

def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET}  {msg}")

def section(title: str) -> None:
    print(f"\n  {DIM}{'─'*50}{RESET}")
    print(f"  {BOLD}{title}{RESET}")
    print(f"  {DIM}{'─'*50}{RESET}")


# ─────────────────────────────────────────────────────────────
#  STEP 1 — Load & audit data
# ─────────────────────────────────────────────────────────────

def load_and_audit(path: str) -> pd.DataFrame:
    banner("STEP 1 · Data loading & audit")

    df = pd.read_csv(path)
    info(f"Raw shape          : {df.shape[0]:,} rows × {df.shape[1]} cols")

    missing = df.isnull().sum().sum()
    dups    = df.duplicated().sum()

    if missing:
        warn(f"Missing values     : {missing:,}  → imputing with 0")
        df = df.fillna(0)
    else:
        success("No missing values")

    if dups:
        warn(f"Duplicate rows     : {dups:,}  ({dups/len(df)*100:.1f} %)  → dropping")
        df = df.drop_duplicates()
        success(f"After dedup shape  : {df.shape[0]:,} rows × {df.shape[1]} cols")
    else:
        success("No duplicate rows")

    n_classes = df["prognosis"].nunique()
    vc        = df["prognosis"].value_counts()
    info(f"Disease classes    : {n_classes}")
    info(f"Samples per class  : min={vc.min()}  max={vc.max()}  mean={vc.mean():.1f}")

    X = df.drop("prognosis", axis=1)
    sparsity = (X == 0).sum().sum() / (X.shape[0] * X.shape[1])
    info(f"Feature sparsity   : {sparsity*100:.1f} % zeros (binary features)")

    return df


# ─────────────────────────────────────────────────────────────
#  STEP 2 — Prepare features
# ─────────────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame):
    banner("STEP 2 · Feature preparation")

    X = df.drop("prognosis", axis=1)
    y = df["prognosis"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    info(f"Feature count      : {X.shape[1]}")
    info(f"Target classes     : {len(le.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=0.2,
        random_state=42,
        stratify=y_enc,
    )
    success(f"Train set          : {X_train.shape[0]:,} samples")
    success(f"Test  set          : {X_test.shape[0]:,} samples")

    return X, y_enc, X_train, X_test, y_train, y_test, le


# ─────────────────────────────────────────────────────────────
#  STEP 3 — Train & cross-validate models
# ─────────────────────────────────────────────────────────────

def train_models(X_train, y_train, X_test, y_test):
    banner("STEP 3 · Model training & cross-validation (5-fold stratified)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Random Forest": CalibratedClassifierCV(
            RandomForestClassifier(
                n_estimators=200,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            method="isotonic", cv=3,
        ),
        "SVM (linear)": CalibratedClassifierCV(
            SVC(
                kernel="linear",
                class_weight="balanced",
                random_state=42,
            ),
            method="sigmoid", cv=3,
        ),
    }

    header = f"  {'Model':<20} {'CV Acc':>8} {'±':>6} {'Test Acc':>10} {'F1 Macro':>10} {'Time':>8}"
    print(f"\n{DIM}{header}{RESET}")
    print(f"  {'─'*65}")

    results = {}
    for name, model in models.items():
        t0       = time.time()
        cv_sc    = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        model.fit(X_train, y_train)
        preds    = model.predict(X_test)
        test_acc = accuracy_score(y_test, preds)
        f1       = f1_score(y_test, preds, average="macro")
        elapsed  = time.time() - t0

        color = GREEN if test_acc >= 0.99 else YELLOW
        print(
            f"  {BOLD}{name:<20}{RESET}"
            f"  {color}{cv_sc.mean():.4f}{RESET}"
            f"  {DIM}±{cv_sc.std():.4f}{RESET}"
            f"  {color}{test_acc:>10.4f}{RESET}"
            f"  {f1:>10.4f}"
            f"  {elapsed:>6.1f}s"
        )
        results[name] = {
            "model":    model,
            "cv_mean":  cv_sc.mean(),
            "cv_std":   cv_sc.std(),
            "test_acc": test_acc,
            "f1":       f1,
        }

    return results


# ─────────────────────────────────────────────────────────────
#  STEP 4 — Evaluate best model
# ─────────────────────────────────────────────────────────────

def evaluate(rf_model, svm_model, X_test, y_test, le):
    banner("STEP 4 · Evaluation — Random Forest (primary)")

    preds = rf_model.predict(X_test)
    print()
    print(classification_report(
        y_test, preds,
        target_names=le.classes_,
        zero_division=0,
    ))

    # Partial-input simulation
    section("Accuracy by number of symptoms entered (SVM vs RF)")
    X_test_df = X_test.copy()
    rng = np.random.default_rng(42)

    print(f"  {'Symptoms':>10}  {'RF Acc':>10}  {'SVM Acc':>10}  {'Winner':>10}")
    print(f"  {'─'*46}")
    for n in [1, 2, 3, 5, "all"]:
        X_partial = X_test_df.copy()
        if n != "all":
            for i in range(len(X_partial)):
                active = X_partial.columns[X_partial.iloc[i] == 1].tolist()
                keep   = rng.choice(active, min(n, len(active)), replace=False).tolist()
                X_partial.iloc[i]      = 0
                X_partial.loc[X_partial.index[i], keep] = 1

        rf_acc  = accuracy_score(y_test, rf_model.predict(X_partial))
        svm_acc = accuracy_score(y_test, svm_model.predict(X_partial))
        winner  = "RF" if rf_acc >= svm_acc else "SVM"
        winner_color = GREEN if winner == "RF" else YELLOW
        print(
            f"  {str(n):>10}  {rf_acc:>10.4f}  {svm_acc:>10.4f}"
            f"  {winner_color}{winner:>10}{RESET}"
        )


# ─────────────────────────────────────────────────────────────
#  STEP 4b — Find optimal RF/SVM crossover threshold
# ─────────────────────────────────────────────────────────────

def find_threshold(rf_model, svm_model, X_test, y_test) -> int:
    """
    Simulate predictions at each symptom count (1..10) and return
    the lowest n where RF accuracy >= SVM accuracy (the crossover point).
    Defaults to 4 if no crossover is found.
    """
    section("Auto-detecting optimal RF/SVM threshold")
    X_test_df = X_test.copy()
    rng = np.random.default_rng(42)
    threshold = 4  # fallback

    print(f"  {'N':>4}  {'RF Acc':>10}  {'SVM Acc':>10}  {'Winner':>10}")
    print(f"  {'─'*40}")
    for n in range(1, 11):
        X_partial = X_test_df.copy()
        for i in range(len(X_partial)):
            active = X_partial.columns[X_partial.iloc[i] == 1].tolist()
            keep   = rng.choice(active, min(n, len(active)), replace=False).tolist()
            X_partial.iloc[i] = 0
            X_partial.loc[X_partial.index[i], keep] = 1

        rf_acc  = accuracy_score(y_test, rf_model.predict(X_partial))
        svm_acc = accuracy_score(y_test, svm_model.predict(X_partial))
        winner  = "RF" if rf_acc >= svm_acc else "SVM"
        color   = GREEN if winner == "RF" else YELLOW
        print(f"  {n:>4}  {rf_acc:>10.4f}  {svm_acc:>10.4f}  {color}{winner:>10}{RESET}")

        if rf_acc >= svm_acc and threshold == 4:
            threshold = n  # first n where RF wins

    success(f"Optimal threshold  : {threshold} symptoms  (RF used when n >= {threshold})")
    return threshold


# ─────────────────────────────────────────────────────────────
#  STEP 5 — Save pipeline
# ─────────────────────────────────────────────────────────────

def save_pipeline(
    rf_model,
    svm_model,
    le: LabelEncoder,
    all_symptoms: list,
    results: dict,
    output_path: str,
    threshold: int = 4,
):
    banner("STEP 5 · Saving pipeline")

    # CalibratedClassifierCV wraps the base estimator — average importances across folds
    try:
        raw_imp = np.mean([
            est.estimator.feature_importances_
            for est in rf_model.calibrated_classifiers_
        ], axis=0)
    except AttributeError:
        raw_imp = rf_model.feature_importances_
    importances = dict(zip(all_symptoms, raw_imp))

    pipeline = {
        "rf_model":       rf_model,
        "svm_model":      svm_model,
        "label_encoder":  le,
        "all_symptoms":   all_symptoms,
        "importances":    importances,
        "threshold":      threshold,
        "metadata": {
            "rf_cv_acc":   round(results["Random Forest"]["cv_mean"], 6),
            "svm_cv_acc":  round(results["SVM (linear)"]["cv_mean"], 6),
            "rf_test_acc": round(results["Random Forest"]["test_acc"], 6),
            "trained_on":  pd.Timestamp.now().isoformat(),
            "n_classes":   len(le.classes_),
            "n_features":  len(all_symptoms),
            "threshold":   threshold,
        },
    }

    with open(output_path, "wb") as f:
        pickle.dump(pipeline, f)

    # ── Versioned copy ──────────────────────────────────────
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    versioned_path = Path(output_path).with_name(f"medical_model_{ts}.pkl")
    with open(versioned_path, "wb") as f:
        pickle.dump(pipeline, f)

    # Keep only the 3 most recent versioned models
    model_dir = Path(output_path).parent
    old_versions = sorted(model_dir.glob("medical_model_????????_??????.pkl"))
    for old in old_versions[:-3]:
        old.unlink()
        warn(f"Removed old version  : {old.name}")

    size_kb = Path(output_path).stat().st_size / 1024
    success(f"Pipeline saved to  : {output_path}  ({size_kb:.0f} KB)")
    success(f"Versioned copy     : {versioned_path.name}")
    info(f"RF  CV accuracy    : {results['Random Forest']['cv_mean']:.4f}")
    info(f"SVM CV accuracy    : {results['SVM (linear)']['cv_mean']:.4f}")
    info(f"RF/SVM threshold   : {threshold} symptoms")
    info(f"Trained on         : {pipeline['metadata']['trained_on'][:19]}")


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}{CYAN}{'═'*60}{RESET}")
    print(f"{BOLD}{CYAN}  Medicine Recommendation System — Training Pipeline{RESET}")
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}")

    df = load_and_audit(DATASET_PATH)
    X, y_enc, X_train, X_test, y_train, y_test, le = prepare_features(df)
    results = train_models(X_train, y_train, X_test, y_test)

    rf_model  = results["Random Forest"]["model"]
    svm_model = results["SVM (linear)"]["model"]

    evaluate(rf_model, svm_model, X_test, y_test, le)
    threshold = find_threshold(rf_model, svm_model, X_test, y_test)
    save_pipeline(
        rf_model, svm_model, le,
        all_symptoms=X.columns.tolist(),
        results=results,
        output_path=OUTPUT_PATH,
        threshold=threshold,
    )

    banner("Training complete", char="═")
    print(f"  {GREEN}Run  python predict.py  to start the recommendation system.{RESET}\n")


if __name__ == "__main__":
    main()
