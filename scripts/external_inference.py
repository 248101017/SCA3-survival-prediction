#!/usr/bin/env python3
"""
External cohort inference (DeepSurv + RSF)
==========================================
Loads pre-trained DeepSurv and RSF model artifacts, applies them to
an independent external cohort, and exports horizon-specific risk
scores with risk-group assignments.

Usage
-----
    python scripts/external_inference.py
    python scripts/external_inference.py \
        --input data/external_cohort.xlsx \
        --artifacts artifacts \
        --outdir outputs

Outputs
-------
    - outer_risk_predictions.csv : per-patient risk scores at ~5y and
      ~9y for both DeepSurv and RSF, with group labels derived from
      training-set median cut-points.

Dependencies
------------
    numpy, pandas, scikit-learn, joblib, openpyxl, auton-survival
"""

import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════

def parse_args():
    """Parse command-line arguments with sensible defaults."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/external_cohort.xlsx",
                   help="Path to external cohort Excel file")
    p.add_argument("--artifacts", default="artifacts",
                   help="Directory containing model artifacts")
    p.add_argument("--outdir", default="outputs",
                   help="Output directory")
    return p.parse_args()

# Feature contract
ID_COL = "id"
HORIZONS = [5.0, 9.0]

RAW_NUM = [
    "BMI", "disease_duration", "ATXN3_CAG_Long",
    "SARA_Total", "EQ_VAS", "PHQ_Depression", "GAD7_Anxiety",
]
RAW_BIN = [
    "INAS_Muscle_atrophy", "INAS_Fasciculations", "INAS_Sensory_symptoms",
]
RAW_ALL = RAW_NUM + RAW_BIN

PROC_CANONICAL = [
    "BMI", "disease_duration", "ATXN3_CAG_Long",
    "SARA_Total", "EQ_VAS", "PHQ_Depression", "GAD7_Anxiety",
    "INAS_Muscle_atrophy_1", "INAS_Fasciculations_1",
    "INAS_Sensory_symptoms_1",
]

# KNN imputation parameters
KNN_K = 5
KNN_WEIGHTS = "distance"

# ═══════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════

def load_artifacts(artifact_dir: str):
    """Load all required model artifacts from a directory."""
    paths = {
        "deepsurv":  os.path.join(artifact_dir, "deepsurv_dcph_model.pkl"),
        "rsf":       os.path.join(artifact_dir, "rsf_auton_model.pkl"),
        "transform": os.path.join(artifact_dir, "transformer.pkl"),
        "times":     os.path.join(artifact_dir, "times.pkl"),
        "cut5":      os.path.join(artifact_dir, "cutoffs_train_median_req5y.json"),
        "cut9":      os.path.join(artifact_dir, "cutoffs_train_median_req9y.json"),
    }

    for label, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {label} artifact: {p}")

    dcph        = joblib.load(paths["deepsurv"])
    rsf_auton   = joblib.load(paths["rsf"])
    transformer = joblib.load(paths["transform"])

    times = np.sort(np.unique(
        np.asarray(joblib.load(paths["times"]), dtype=float)))

    with open(paths["cut5"], "r", encoding="utf-8") as f:
        cut5 = json.load(f)
    with open(paths["cut9"], "r", encoding="utf-8") as f:
        cut9 = json.load(f)

    for key in ["DeepSurv_cutoff", "RSF_cutoff"]:
        if key not in cut5:
            raise KeyError(f"{paths['cut5']} missing key: {key}")
        if key not in cut9:
            raise KeyError(f"{paths['cut9']} missing key: {key}")

    cutoffs = {
        "deep_5y":  float(cut5["DeepSurv_cutoff"]),
        "rsf_5y":   float(cut5["RSF_cutoff"]),
        "deep_9y":  float(cut9["DeepSurv_cutoff"]),
        "rsf_9y":   float(cut9["RSF_cutoff"]),
        "grid_5y":  float(cut5.get("grid_used_years", float("nan"))),
        "grid_9y":  float(cut9.get("grid_used_years", float("nan"))),
    }

    return dcph, rsf_auton, transformer, times, cutoffs

def ensure_processed_features(X_proc_raw, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the transformer output matches the 10-feature contract
    expected by the survival models.  Handles several edge cases
    (numpy array output, missing one-hot columns, etc.).
    """
    if isinstance(X_proc_raw, np.ndarray):
        X_proc = pd.DataFrame(X_proc_raw)
    elif isinstance(X_proc_raw, pd.DataFrame):
        X_proc = X_proc_raw.copy()
    else:
        X_proc = pd.DataFrame(X_proc_raw)

    # Assign column names if transformer returned unnamed array
    if X_proc.columns is None or any(isinstance(c, int) for c in X_proc.columns):
        if X_proc.shape[1] == 7:
            X_proc.columns = RAW_NUM
        elif X_proc.shape[1] == 10:
            X_proc.columns = PROC_CANONICAL

    cols = list(X_proc.columns)

    # Case 1: already in canonical form
    if all(c in cols for c in PROC_CANONICAL):
        return X_proc[PROC_CANONICAL].copy()

    # Case 2: transformer only scaled numeric features (7 cols)
    if X_proc.shape[1] == 7 and all(c in cols for c in RAW_NUM):
        out = X_proc[RAW_NUM].copy()
        for raw, proc in [
            ("INAS_Muscle_atrophy",   "INAS_Muscle_atrophy_1"),
            ("INAS_Fasciculations",   "INAS_Fasciculations_1"),
            ("INAS_Sensory_symptoms", "INAS_Sensory_symptoms_1"),
        ]:
            out[proc] = raw_df[raw].astype(int).values
        return out[PROC_CANONICAL].copy()

    # Case 3: raw column names (10 cols, not yet renamed)
    if all(c in cols for c in RAW_ALL):
        out = X_proc[RAW_ALL].copy()
        for raw, proc in [
            ("INAS_Muscle_atrophy",   "INAS_Muscle_atrophy_1"),
            ("INAS_Fasciculations",   "INAS_Fasciculations_1"),
            ("INAS_Sensory_symptoms", "INAS_Sensory_symptoms_1"),
        ]:
            out[proc] = out[raw].astype(int)
        out = out.drop(columns=RAW_BIN)
        return out[PROC_CANONICAL].copy()

    # Case 4: fallback — pad missing columns
    out = X_proc.copy()
    for c in PROC_CANONICAL:
        if c not in out.columns:
            if c.endswith("_1"):
                raw_name = c.replace("_1", "")
                out[c] = (raw_df[raw_name].astype(int).values
                          if raw_name in raw_df.columns else 0)
            else:
                out[c] = (raw_df[c].values
                          if c in raw_df.columns else 0)
    return out[PROC_CANONICAL].copy()

# ═══════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    out_csv = os.path.join(args.outdir, "outer_risk_predictions.csv")

    # ── Load artifacts ─────────────────────────────────────
    print("[INFO] Loading artifacts...")
    dcph, rsf_auton, transformer, times, cutoffs = \
        load_artifacts(args.artifacts)

    print(f"[INFO] Time grid: n={len(times)}, "
          f"min={times.min():.3f}, max={times.max():.3f}")
    print(f"[INFO] 5y cutoffs: DeepSurv={cutoffs['deep_5y']:.6f}, "
          f"RSF={cutoffs['rsf_5y']:.6f}, "
          f"grid_used={cutoffs['grid_5y']:.3f}")
    print(f"[INFO] 9y cutoffs: DeepSurv={cutoffs['deep_9y']:.6f}, "
          f"RSF={cutoffs['rsf_9y']:.6f}, "
          f"grid_used={cutoffs['grid_9y']:.3f}")

    # ── Load external cohort ───────────────────────────────
    print(f"[INFO] Loading external cohort: {args.input}")
    df = pd.read_excel(args.input)

    if ID_COL not in df.columns:
        raise ValueError(
            f"External cohort file missing ID column '{ID_COL}'. "
            f"Available columns: {list(df.columns)}")

    missing = [c for c in RAW_ALL if c not in df.columns]
    if missing:
        raise ValueError(
            f"External cohort file missing feature columns: {missing}")

    ids = df[ID_COL].astype(str).copy()
    X = df[RAW_ALL].copy()
    for c in RAW_ALL:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    print(f"[INFO] External cohort: {len(X)} subjects, "
          f"{len(RAW_ALL)} features")

    # ── KNN imputation (external cohort only) ──────────────
    n_missing = X.isna().sum().sum()
    print(f"[INFO] Missing values before imputation: {n_missing}")

    imputer = KNNImputer(n_neighbors=KNN_K, weights=KNN_WEIGHTS)
    X_imp = pd.DataFrame(
        imputer.fit_transform(X), columns=RAW_ALL)

    # Round binary features back to 0/1
    for c in RAW_BIN:
        X_imp[c] = np.clip(np.rint(X_imp[c].values), 0, 1).astype(int)
    for c in RAW_NUM:
        X_imp[c] = X_imp[c].astype(float)

    # ── Transform using training-set transformer ───────────
    X_proc_raw = transformer.transform(X_imp)
    X_proc = ensure_processed_features(X_proc_raw, X_imp)
    print(f"[INFO] Processed feature matrix: {X_proc.shape}")

    # ── Predict survival at all grid points ────────────────
    surv_dcph = np.asarray(
        dcph.predict_survival(X_proc, times), dtype=float)
    surv_rsf = np.asarray(
        rsf_auton.predict_survival(X_proc, times), dtype=float)

    # ── Extract horizon-specific risks ─────────────────────
    out = pd.DataFrame({ID_COL: ids})

    for t0 in HORIZONS:
        t_idx = int(np.argmin(np.abs(times - float(t0))))
        t_used = float(times[t_idx])
        tag = f"{int(t0)}y"

        out[f"t_used_{tag}"] = t_used
        out[f"deepsurv_risk_{tag}"] = 1.0 - surv_dcph[:, t_idx]
        out[f"rsf_risk_{tag}"] = 1.0 - surv_rsf[:, t_idx]

    # ── Assign risk groups (training-set median cut-points) ─
    out["deepsurv_group_5y_median"] = np.where(
        out["deepsurv_risk_5y"] >= cutoffs["deep_5y"], "high", "low")
    out["rsf_group_5y_median"] = np.where(
        out["rsf_risk_5y"] >= cutoffs["rsf_5y"], "high", "low")
    out["deepsurv_group_9y_median"] = np.where(
        out["deepsurv_risk_9y"] >= cutoffs["deep_9y"], "high", "low")
    out["rsf_group_9y_median"] = np.where(
        out["rsf_risk_9y"] >= cutoffs["rsf_9y"], "high", "low")

    # Store cut-points for traceability
    out["deep_cutoff_5y_median"] = cutoffs["deep_5y"]
    out["rsf_cutoff_5y_median"]  = cutoffs["rsf_5y"]
    out["deep_cutoff_9y_median"] = cutoffs["deep_9y"]
    out["rsf_cutoff_9y_median"]  = cutoffs["rsf_9y"]

    # ── Save ───────────────────────────────────────────────
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[DONE] Saved: {out_csv}")
    print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()