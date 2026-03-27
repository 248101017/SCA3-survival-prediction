#!/usr/bin/env python3
"""
SCA3 survival model training and evaluation
============================================
Trains and benchmarks four survival model architectures (CoxPH,
DeepSurv, deep Cox mixtures, random survival forest) using the
auton_survival framework.  Generates all figures and tables
reported in the manuscript.

Usage
-----
    python scripts/model_training.py
    python scripts/model_training.py \
        --input data/processed_cohort.csv \
        --outdir outputs \
        --artifacts artifacts

Outputs (in --outdir)
---------------------
    - Time grid:           times_grid.csv / .txt
    - Per-model metrics:   metrics_{model}.csv / .txt
    - Summary table:       model_best_params.csv / .md
    - Environment:         environment.txt
    - Figures:             AUC(t), C-index(t), IBS, ROC overlays,
                           calibration, DCA, KM stratification,
                           bootstrap CI, SHAP, permutation importance

Dependencies
------------
    numpy, pandas, matplotlib, seaborn, scikit-learn, scikit-survival,
    lifelines, shap, joblib, auton-survival
"""

from __future__ import annotations

import os
import sys
import platform
import warnings
import argparse
from datetime import datetime
from typing import Tuple, List, Dict, Optional

# Must be set before importing pyplot
import matplotlib as mpl
mpl.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import auc
from sklearn.inspection import permutation_importance

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════
# Command-line arguments
# ═══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input",
                   default="data/processed_cohort.csv",
                   help="Path to processed CSV (imputed, LASSO-selected)")
    p.add_argument("--outdir",
                   default="outputs",
                   help="Output directory for figures and tables")
    p.add_argument("--artifacts",
                   default="artifacts",
                   help="Directory to export Streamlit artifacts")
    p.add_argument("--seed", type=int, default=1,
                   help="Random seed for reproducibility")
    return p.parse_args()

# ═══════════════════════════════════════════════════════════
# Plotting style (npj Digital Medicine compatible)
# ═══════════════════════════════════════════════════════════

FONT_SANS = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]

def set_journal_style():
    """Set matplotlib defaults for npj Digital Medicine figures."""
    mpl.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   FONT_SANS,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
        "axes.unicode_minus": False,
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "axes.linewidth":    1.0,
        "lines.linewidth":   2.0,
        "grid.linewidth":    0.8,
        "figure.dpi":        150,
        "savefig.dpi":       300,
    })
    sns.set_theme(style="white")

def try_register_arial():
    """Best-effort registration of Arial font on Windows."""
    try:
        import matplotlib.font_manager as fm
        candidates = [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\arialbd.ttf",
        ]
        for fp in candidates:
            if os.path.exists(fp):
                fm.fontManager.addfont(fp)
                break
    except Exception:
        pass

def force_axes_font(ax, family="Arial"):
    """Force a specific font family on an existing axes object."""
    ax.title.set_fontfamily(family)
    ax.xaxis.label.set_fontfamily(family)
    ax.yaxis.label.set_fontfamily(family)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontfamily(family)

try_register_arial()
set_journal_style()

# ═══════════════════════════════════════════════════════════
# Colour palette (colour-blind friendly)
# ═══════════════════════════════════════════════════════════

COLOR_CPH = "#2ca02c"      # green  – CoxPH
COLOR_DS  = "#1f77b4"      # blue   – DeepSurv
COLOR_DCM = "#ff7f0e"      # orange – DCM
COLOR_RSF = "#d62728"      # red    – RSF
COLOR_NB  = "#6a3d9a"      # purple – net benefit difference
SCI_PALETTE = [COLOR_DS, COLOR_DCM, COLOR_CPH, COLOR_RSF]

SHAP_CMAP = plt.get_cmap("viridis")

# Academic display names for SHAP and permutation importance
FEATURE_DISPLAY_NAME = {
    "BMI":                     "BMI",
    "disease_duration":        "Disease duration",
    "ATXN3_CAG_Long":          "Long CAG repeats",
    "SARA_Total":              "SARA score",
    "EQ_VAS":                  "EQ-VAS",
    "PHQ_Depression":          "PHQ-9 depression",
    "GAD7_Anxiety":            "GAD-7 anxiety",
    "INAS_Muscle_atrophy_1":   "INAS muscle atrophy",
    "INAS_Fasciculations_1":   "INAS fasciculations",
    "INAS_Sensory_symptoms_1": "INAS sensory symptoms",
}

# ═══════════════════════════════════════════════════════════
# Pipeline configuration
# ═══════════════════════════════════════════════════════════

# Column names
ID_COL = "id"

# Time grid construction
N_QUANTILE_POINTS = 10
QUANTILE_MIN = 0.1
QUANTILE_MAX = 1.0

# Evaluation horizons (years)
EVAL_HORIZONS_REQ = [5.0, 9.0]

# Toggle blocks
RUN_DS_VS_RSF_OVERLAYS     = True
RUN_DS_VS_RSF_KM_PANEL     = True
RUN_DS_VS_RSF_NB_DIFF      = True

RUN_DS_RSF_TEST_STRAT_PRED_SURV_PANEL_TRAIN_MEDIAN      = True
RUN_DS_RSF_TEST_STRAT_PRED_SURV_PANEL_TRAIN_IPCW_YOUDEN = True
PRED_SURV_PANEL_HORIZONS_REQ = [5.0, 9.0]

# IPCW-Youden cut-point selection
IPCW_YOUDEN_N_THRESHOLDS = 200

# Bootstrap (test set, stratified by event status)
RUN_BOOTSTRAP    = True
BOOTSTRAP_B      = 1000
BOOTSTRAP_SEED   = 123
BOOTSTRAP_ALPHA  = 0.05

# RSF permutation importance (scikit-survival)
RUN_RSF_PERM_IMPORTANCE       = True
PERM_IMPORTANCE_HORIZONS_REQ  = [5.0, 9.0]
PERM_IMPORTANCE_N_REPEATS     = 200
PERM_IMPORTANCE_RANDOM_SEED   = 2026

# SHAP (DeepSurv, Kernel SHAP)
RUN_SHAP            = True
SHAP_DEP_PLOTS      = 10
SHAP_HORIZONS_REQ   = [5.0, 9.0]
SHAP_BG_N           = 50
SHAP_NSAMPLES       = 250
SHAP_BATCH_SIZE     = 25
SHAP_SUPTITLE_FS    = 22
SHAP_SUPTITLE_FW    = "bold"

# Combined figure export toggles
EXPORT_SHAP_5Y_9Y_PANEL     = True
EXPORT_PERMIMP_5Y_9Y_PANEL  = True

DATASET_LABEL = "Test set"
EXPORT_ARTIFACTS = True

# ═══════════════════════════════════════════════════════════
# auton_survival imports
# ═══════════════════════════════════════════════════════════

from auton_survival.preprocessing import Preprocessor
from auton_survival.estimators import SurvivalModel
from auton_survival.metrics import survival_regression_metric

# ═══════════════════════════════════════════════════════════
# Title helpers (centralised)
# ═══════════════════════════════════════════════════════════

def _h_label(h: float) -> str:
    """Format a horizon as a compact label, e.g. 't approx 5y'."""
    return f"t\u22485{int(round(float(h)))}y"

def _cut_label(cutoff_type: str) -> str:
    """Return a human-readable cut-point description."""
    ct = str(cutoff_type).strip().lower()
    if ct == "median":
        return "Training median cut-point"
    if ct in ("ipcw_youden", "youden", "ipcw-youden"):
        return "Training IPCW\u2013Youden cut-point"
    return cutoff_type

def sci_title(kind: str, *,
              horizon: Optional[float] = None,
              cutoff_type: Optional[str] = None) -> str:
    """Generate a standardised figure title string."""
    h   = None if horizon is None else _h_label(horizon)
    ds  = DATASET_LABEL
    cut = None if cutoff_type is None else _cut_label(cutoff_type)

    titles = {
        "auc_panel_A":  f"a) Time-dependent discrimination: AUC(t) ({ds})",
        "ctd_panel_B":  f"b) Time-dependent discrimination: C-index(t) ({ds})",
        "ibs_panel_C":  f"c) Overall prediction error: IBS ({ds})",
        "auc_line":     f"Time-dependent AUC(t) ({ds})",
        "ctd_line":     f"Time-dependent C-index(t) ({ds})",
        "ibs_dot":      f"Integrated Brier Score (IBS) ({ds})",
        "roc_overlay":  f"Time-dependent ROC at t\u22485y and t\u22489y ({ds})",
        "cal_overlay":  f"Calibration at t\u22485y and t\u22489y ({ds})",
        "dca":          f"Decision curve analysis at {h} ({ds})",
        "combined_2x2": f"DeepSurv versus RSF: discrimination, calibration, "
                        f"and clinical utility ({ds})",
        "combined_A":   "a) ROC at t\u22485y and t\u22489y",
        "combined_B":   "b) Calibration at t\u22485y and t\u22489y",
        "combined_C":   "c) Decision curve analysis at t\u22485y",
        "combined_D":   "d) Decision curve analysis at t\u22489y",
        "nb_diff":      f"Net benefit difference: DeepSurv minus RSF ({ds})",
        "nb_diff_panel_A": "a) Net benefit difference at t\u22485y",
        "nb_diff_panel_B": "b) Net benefit difference at t\u22489y",
        "km_overall":   f"Risk stratification using training-derived "
                        f"cut-points ({ds})",
        "km_A_deepsurv": "a) DeepSurv: Kaplan\u2013Meier survival by risk group",
        "km_B_rsf":      "b) RSF: Kaplan\u2013Meier survival by risk group",
        "predsurv_overall": f"Predicted survival by risk group at {h} "
                           f"({ds}); {cut}",
        "predsurv_A_deepsurv": f"a) DeepSurv: Predicted survival at {h} ({ds})",
        "predsurv_B_rsf":      f"b) RSF: Predicted survival at {h} ({ds})",
        "boot_auc":     f"Bootstrap 95% CI for AUC(t) ({ds})",
        "boot_ctd":     f"Bootstrap 95% CI for C-index(t) ({ds})",
        "boot_2panel":  f"Bootstrap uncertainty for time-dependent "
                        f"discrimination ({ds})",
        "boot_2panel_A": "a) AUC(t) with 95% CI",
        "boot_2panel_B": "b) C-index(t) with 95% CI",
        "perm_importance": f"RSF permutation importance at {h} ({ds})",
        "perm_panel":   f"RSF permutation importance at t\u22485y and "
                        f"t\u22489y ({ds})",
        "shap":         f"DeepSurv SHAP values at {h} ({ds})",
        "shap_panel":   f"DeepSurv SHAP values at t\u22485y and t\u22489y "
                        f"({ds})",
    }
    return titles.get(kind, kind)

# ═══════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════

def savefig(name: str, output_dir: str = "outputs"):
    """Save current figure as PNG and PDF."""
    out = os.path.join(output_dir, name)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    root, ext = os.path.splitext(out)
    if ext.lower() == ".png":
        out_pdf = root + ".pdf"
        plt.savefig(out_pdf, bbox_inches="tight")
        print(f"Saved: {out_pdf}")
    plt.close()

def ensure_dataframe(X) -> pd.DataFrame:
    """Convert array-like to DataFrame if needed."""
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(np.asarray(X))

def nearest_time_idx(times_arr, t_req: float) -> Tuple[int, float]:
    """Return index and value of the time grid point nearest to t_req."""
    arr = np.asarray(times_arr, dtype=float)
    idx = int(np.argmin(np.abs(arr - float(t_req))))
    return idx, float(arr[idx])

def predict_survival(model, X: pd.DataFrame,
                     times_list) -> np.ndarray:
    """Predict survival probabilities for all subjects at all times."""
    return np.asarray(
        model.predict_survival(X, times_list), dtype=float)

def fit_censoring_km(time, event):
    """Fit a Kaplan-Meier estimator for the censoring distribution."""
    km = KaplanMeierFitter()
    km.fit(time, event_observed=1 - event)
    return km

def ipcw_roc(time, event, risk, t0, km_censor,
             thresholds=np.linspace(0, 1, 200)):
    """
    Compute IPCW-weighted ROC curve at a fixed time horizon.

    Returns FPR, TPR, and the threshold array.
    """
    TPR, FPR = [], []
    for c in thresholds:
        tp = fp = fn = tn = 0.0
        treat = risk >= c
        for i in range(len(time)):
            if time[i] <= t0 and event[i] == 1:
                w = 1.0 / (km_censor.predict(time[i]) + 1e-12)
                tp += w * treat[i]
                fn += w * (1 - treat[i])
            elif time[i] > t0:
                w = 1.0 / (km_censor.predict(t0) + 1e-12)
                fp += w * treat[i]
                tn += w * (1 - treat[i])
        TPR.append(tp / (tp + fn + 1e-8))
        FPR.append(fp / (fp + tn + 1e-8))
    return np.array(FPR), np.array(TPR), thresholds

def stdca_ipcw(time, event, risk, t0, thresholds):
    """IPCW-corrected standardised net benefit for DCA."""
    km_censor = KaplanMeierFitter()
    km_censor.fit(time, event_observed=1 - event)
    n = len(time)
    net_benefit = []
    for pt in thresholds:
        TP = FP = 0.0
        treat = risk >= pt
        for j in range(n):
            if time[j] <= t0 and event[j] == 1:
                w = 1.0 / (km_censor.predict(time[j]) + 1e-12)
                TP += w * treat[j]
            elif time[j] > t0:
                w = 1.0 / (km_censor.predict(t0) + 1e-12)
                FP += w * treat[j]
        nb = (TP / n) - (FP / n) * (pt / (1 - pt))
        net_benefit.append(nb)
    return np.array(net_benefit)

def treat_all_nb(time, event, t0, thresholds):
    """Net benefit of a treat-all strategy at each threshold."""
    km = KaplanMeierFitter()
    km.fit(time, event_observed=event)
    event_rate = 1 - km.predict(t0)
    return np.array([
        event_rate - (pt / (1 - pt)) * (1 - event_rate)
        for pt in thresholds
    ])

def treat_none_nb(thresholds):
    """Net benefit of a treat-none strategy (always zero)."""
    return np.zeros_like(thresholds)

def panel_from_two_images(
    img_left_path: str,
    img_right_path: str,
    out_path: str,
    suptitle: str,
    subtitles: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (20, 9),
    suptitle_fs: int = 20,
    subtitle_fs: int = 14,
    top_rect: float = 0.94,
    wspace: float = 0.06,
    title_pad: int = 10,
):
    """Compose a 1x2 panel figure from two pre-rendered images."""
    set_journal_style()

    if not (os.path.exists(img_left_path)
            and os.path.exists(img_right_path)):
        print(f"[WARN] Panel skipped (missing images): "
              f"{img_left_path}, {img_right_path}")
        return

    if subtitles is None:
        subtitles = ["", ""]

    left  = mpimg.imread(img_left_path)
    right = mpimg.imread(img_right_path)

    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor="white")
    fig.suptitle(suptitle, fontsize=suptitle_fs, fontweight="bold",
                 fontfamily="Arial", y=0.985)

    for ax, img, letter, sub in zip(
            axes, [left, right], ["a)", "b)"], subtitles):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{letter} {sub}".strip(),
                     fontsize=subtitle_fs, pad=title_pad,
                     fontweight="bold", fontfamily="Arial", loc="center")

    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01,
                        top=top_rect, wspace=wspace)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")

def panel_2x2_from_images(
    img_tl: str, img_tr: str, img_bl: str, img_br: str,
    out_path: str,
    suptitle: str,
    subtitles: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (20, 18),
    suptitle_fs: int = 20,
    subtitle_fs: int = 14,
    top_rect: float = 0.955,
    wspace: float = 0.06,
    hspace: float = 0.08,
    title_pad: int = 10,
):
    """Compose a 2x2 panel figure from four pre-rendered images."""
    set_journal_style()

    paths = [img_tl, img_tr, img_bl, img_br]
    if not all(os.path.exists(p) for p in paths):
        print(f"[WARN] 2x2 panel skipped (missing images)")
        return

    if subtitles is None:
        subtitles = ["", "", "", ""]

    imgs = [mpimg.imread(p) for p in paths]
    letters = ["a)", "b)", "c)", "d)"]

    fig, axes = plt.subplots(2, 2, figsize=figsize, facecolor="white")
    fig.suptitle(suptitle, fontsize=suptitle_fs, fontweight="bold",
                 fontfamily="Arial", y=0.985)

    for ax, img, letter, sub in zip(
            axes.flat, imgs, letters, subtitles):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{letter} {sub}".strip(),
                     fontsize=subtitle_fs, pad=title_pad,
                     fontweight="bold", fontfamily="Arial", loc="center")

    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01,
                        top=top_rect, wspace=wspace, hspace=hspace)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")

# ============================================================
# 5) Load data
# ============================================================
data_path = os.path.join(args.input)
if not os.path.exists(data_path) and os.path.exists(data_path + ".csv"):
    data_path = data_path + ".csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"Cannot find data file: {args.input} "
        f"(also tried: {args.input + '.csv'})")

df = pd.read_csv(data_path)

if ID_COL not in df.columns:
    raise ValueError(
        f"ID column '{ID_COL}' not found. "
        f"Available columns: {list(df.columns)}")
if "duration" not in df.columns or "event" not in df.columns:
    raise ValueError(
        "Input CSV must include columns: 'duration' and 'event'.")

outcomes = df[[ID_COL, "duration", "event"]].rename(
    columns={"duration": "time"})
features = df.drop(columns=[ID_COL, "duration", "event"])

cat_feats = [
    "INAS_Muscle_atrophy",
    "INAS_Fasciculations",
    "INAS_Sensory_symptoms",
]
num_feats = [
    "BMI", "disease_duration", "ATXN3_CAG_Long",
    "SARA_Total", "EQ_VAS", "PHQ_Depression", "GAD7_Anxiety",
]

# ============================================================
# 6) Train / validation / test split
# ============================================================
x_tr, x_te, y_tr, y_te = train_test_split(
    features, outcomes,
    test_size=0.2,
    random_state=args.seed,
    stratify=outcomes["event"],
)
x_tr, x_val, y_tr, y_val = train_test_split(
    x_tr, y_tr,
    test_size=0.25,
    random_state=args.seed,
    stratify=y_tr["event"],
)

y_tr_model  = y_tr[["time", "event"]].copy()
y_val_model = y_val[["time", "event"]].copy()
y_te_model  = y_te[["time", "event"]].copy()

# Retain raw training features for artifact export
x_tr_raw = x_tr.copy()

print(f"[INFO] Split sizes: train={len(x_tr)}, "
      f"val={len(x_val)}, test={len(x_te)}")

# ============================================================
# 7) Feature preprocessing (auton_survival Preprocessor)
# ============================================================
preprocessor = Preprocessor(
    cat_feat_strat="ignore",
    num_feat_strat="mean",
)
transformer = preprocessor.fit(
    x_tr,
    cat_feats=cat_feats,
    num_feats=num_feats,
    one_hot=True,
    fill_value=-1,
)

x_tr  = ensure_dataframe(transformer.transform(x_tr))
x_val = ensure_dataframe(transformer.transform(x_val))
x_te  = ensure_dataframe(transformer.transform(x_te))

def sanitize_features(
    X_tr: pd.DataFrame,
    X_val: pd.DataFrame,
    X_te: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Replace inf with NaN, then fill NaN using training-set
    column means.  Ensures all three splits share the same
    column set and contain no non-finite values.
    """
    X_tr  = X_tr.copy()
    X_val = X_val.copy()
    X_te  = X_te.copy()

    for X in (X_tr, X_val, X_te):
        X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Compute fill values from training set only
    fill: Dict[str, float] = {}
    for c in X_tr.columns:
        v = pd.to_numeric(X_tr[c], errors="coerce").to_numpy(dtype=float)
        m = np.nanmean(v)
        fill[c] = float(m) if np.isfinite(m) else 0.0

    X_tr  = X_tr.fillna(value=fill)
    X_val = (X_val.reindex(columns=X_tr.columns, fill_value=np.nan)
                  .fillna(value=fill))
    X_te  = (X_te.reindex(columns=X_tr.columns, fill_value=np.nan)
                 .fillna(value=fill))

    # Final validation
    for label, X in [("train", X_tr), ("val", X_val), ("test", X_te)]:
        arr = X.to_numpy(dtype=float)
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError(
                f"Sanitize failed: {label} still contains NaN/inf.")

    print("[INFO] Feature sanitisation complete (no NaN/inf).")
    return X_tr, X_val, X_te

x_tr, x_val, x_te = sanitize_features(x_tr, x_val, x_te)

# ============================================================
# 7.5) Build evaluation time grid from training event times
# ============================================================
event_times = (y_tr_model
               .loc[y_tr_model["event"] == 1, "time"]
               .to_numpy(dtype=float))
if event_times.size == 0:
    raise ValueError(
        "No events in training set; cannot build time grid.")

q_grid = np.linspace(
    float(QUANTILE_MIN), float(QUANTILE_MAX),
    int(N_QUANTILE_POINTS))
times = sorted(set(
    float(t) for t in np.quantile(event_times, q_grid)))

print(f"\n[INFO] Evaluation time grid ({len(times)} points):")
for i, t in enumerate(times, 1):
    print(f"  t{i:02d} = {t:.4f} years")

times_df = pd.DataFrame({
    "idx": np.arange(1, len(times) + 1),
    "time_years": times,
})
times_csv = os.path.join(args.outdir, "times_grid.csv")
times_txt = os.path.join(args.outdir, "times_grid.txt")
times_df.to_csv(times_csv, index=False, encoding="utf-8-sig")
with open(times_txt, "w", encoding="utf-8") as f:
    for t in times:
        f.write(f"{t:.10f}\n")
print(f"[INFO] Saved: {times_csv}")
print(f"[INFO] Saved: {times_txt}")

# ============================================================
# 8) Model training and evaluation
# ============================================================

def align_xy(
    X: pd.DataFrame, y: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reset indices and enforce float dtype for model input."""
    X2 = X.reset_index(drop=True).copy()
    y2 = y.reset_index(drop=True).copy()
    X2[:] = X2.to_numpy(dtype=float)
    if len(X2) != len(y2):
        raise ValueError(
            f"X/y length mismatch: X={len(X2)}, y={len(y2)}")
    return X2, y2

def train_and_evaluate(
    model_name: str,
    param_grid: dict,
    x_tr: pd.DataFrame,
    y_tr: pd.DataFrame,
    x_val: pd.DataFrame,
    y_val: pd.DataFrame,
    x_te: pd.DataFrame,
    y_te: pd.DataFrame,
    times: list,
) -> Tuple:
    """
    Grid-search over param_grid, select the model with lowest
    validation IBS, and return test-set metrics.

    Returns
    -------
    best_model : SurvivalModel
    results : dict  (AUC, IBS, Concordance Index, best_param,
                     best_val_ibs)
    pred_te : np.ndarray  (test-set survival predictions)
    pred_tr : np.ndarray  (train-set survival predictions)
    """
    x_tr2,  y_tr2  = align_xy(x_tr, y_tr)
    x_val2, y_val2 = align_xy(x_val, y_val)
    x_te2,  y_te2  = align_xy(x_te, y_te)

    candidates = []
    for param in ParameterGrid(param_grid):
        model = SurvivalModel(model_name, **param)
        model.fit(x_tr2, y_tr2)
        pred_val = model.predict_survival(x_val2, times)
        ibs_val = survival_regression_metric(
            "ibs", y_val2, pred_val, times, y_tr2)
        candidates.append((ibs_val, model, param))

    best_ibs, best_model, best_param = sorted(
        candidates, key=lambda x: x[0])[0]

    pred_te = best_model.predict_survival(x_te2, times)
    pred_tr = best_model.predict_survival(x_tr2, times)

    results = {
        "AUC": survival_regression_metric(
            "auc", y_te2, pred_te, times, y_tr2),
        "IBS": survival_regression_metric(
            "ibs", y_te2, pred_te, times, y_tr2),
        "Concordance Index": survival_regression_metric(
            "ctd", y_te2, pred_te, times, y_tr2),
        "best_param": best_param,
        "best_val_ibs": best_ibs,
    }
    return best_model, results, pred_te, pred_tr

# ============================================================
# 9) Train all four survival models
# ============================================================
print("\n[INFO] Training CoxPH...")
model_cph, res_cph, pred_te_cph, pred_tr_cph = train_and_evaluate(
    "cph",
    {"l2": [1e-3, 1e-4]},
    x_tr, y_tr_model, x_val, y_val_model,
    x_te, y_te_model, times,
)

print("[INFO] Training DeepSurv (DCPH)...")
model_dcph, res_dcph, pred_te_dcph, pred_tr_dcph = train_and_evaluate(
    "dcph",
    {
        "bs": [100, 200],
        "learning_rate": [1e-4, 1e-3],
        "layers": [[100], [100, 100]],
    },
    x_tr, y_tr_model, x_val, y_val_model,
    x_te, y_te_model, times,
)

print("[INFO] Training Deep Cox Mixtures (DCM)...")
model_dcm, res_dcm, pred_te_dcm, pred_tr_dcm = train_and_evaluate(
    "dcm",
    {
        "k": [2, 3],
        "learning_rate": [1e-3, 1e-4],
        "layers": [[100], [100, 100]],
    },
    x_tr, y_tr_model, x_val, y_val_model,
    x_te, y_te_model, times,
)

print("[INFO] Training Random Survival Forest (RSF)...")
model_rsf, res_rsf, pred_te_rsf, pred_tr_rsf = train_and_evaluate(
    "rsf",
    {
        "n_estimators": [100, 200],
        "max_depth": [10, 20],
        "min_samples_split": [2, 5],
    },
    x_tr, y_tr_model, x_val, y_val_model,
    x_te, y_te_model, times,
)

print("\n[INFO] Best hyperparameters (selected by validation IBS):")
for name, res in [("CoxPH", res_cph), ("DeepSurv", res_dcph),
                  ("DCM", res_dcm), ("RSF", res_rsf)]:
    print(f"  {name}: {res['best_param']}  "
          f"(val IBS = {res['best_val_ibs']:.4f})")

# ============================================================
# 10) Export metrics, parameter table, and environment info
# ============================================================

def to_float_list(x) -> List[float]:
    """Convert array-like to a flat list of floats."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    return [float(v) for v in arr]

def export_model_metrics(
    output_dir: str,
    model_name: str,
    res: dict,
    times: List[float],
):
    """Save per-model time-dependent metrics to CSV and TXT."""
    auc_list = to_float_list(res.get("AUC", []))
    ctd_list = to_float_list(res.get("Concordance Index", []))
    ibs_val  = float(
        np.asarray(res.get("IBS", np.nan)).reshape(-1)[0])

    n = len(times)
    dfm = pd.DataFrame({
        "time_years": [float(t) for t in times],
        "auc": auc_list if len(auc_list) == n else [np.nan] * n,
        "ctd": ctd_list if len(ctd_list) == n else [np.nan] * n,
    })
    dfm["ibs"] = ibs_val

    out_csv = os.path.join(output_dir, f"metrics_{model_name}.csv")
    out_txt = os.path.join(output_dir, f"metrics_{model_name}.txt")
    dfm.to_csv(out_csv, index=False, encoding="utf-8-sig")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"=== Metrics: {model_name} ===\n")
        f.write(f"Generated: "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"IBS (scalar) = {ibs_val:.6f}\n\n")
        f.write("Time-dependent metrics:\n")
        for i, t in enumerate(times):
            f.write(f"  t={t:.4f}  "
                    f"AUC={dfm.loc[i, 'auc']:.4f}  "
                    f"C-index={dfm.loc[i, 'ctd']:.4f}\n")

    print(f"[INFO] Saved: {out_csv}")
    print(f"[INFO] Saved: {out_txt}")

export_model_metrics(args.outdir, "cph",          res_cph,  times)
export_model_metrics(args.outdir, "deepsurv_dcph", res_dcph, times)
export_model_metrics(args.outdir, "dcm",          res_dcm,  times)
export_model_metrics(args.outdir, "rsf_auton",    res_rsf,  times)

def format_param_dict(d: dict) -> str:
    """Format a parameter dictionary as a semicolon-separated string."""
    return "; ".join(f"{k}={d[k]}" for k in sorted(d))

def safe_mean(x) -> float:
    """Compute mean of array-like, returning NaN if empty."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    return float(np.nanmean(arr)) if arr.size else float("nan")

def df_to_markdown(df_: pd.DataFrame) -> str:
    """Convert a DataFrame to a Markdown table string."""
    cols = list(df_.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df_.iterrows():
        lines.append(
            "| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"

def export_best_params_table(output_dir: str):
    """Export a summary table of best hyperparameters and metrics."""
    rows = []
    for label, res in [
        ("CoxPH (cph)",     res_cph),
        ("DeepSurv (dcph)", res_dcph),
        ("DCM (dcm)",       res_dcm),
        ("RSF (rsf)",       res_rsf),
    ]:
        rows.append({
            "model":         label,
            "best_param":    format_param_dict(
                res.get("best_param", {})),
            "val_IBS":       float(res.get("best_val_ibs")),
            "test_AUC_mean": safe_mean(res.get("AUC")),
            "test_IBS":      float(
                np.asarray(res.get("IBS")).reshape(-1)[0]),
            "test_CTD_mean": safe_mean(
                res.get("Concordance Index")),
        })

    df_params = pd.DataFrame(rows)
    for c in ["val_IBS", "test_AUC_mean",
              "test_IBS", "test_CTD_mean"]:
        df_params[c] = df_params[c].map(
            lambda v: f"{float(v):.4f}")

    out_csv = os.path.join(output_dir, "model_best_params.csv")
    df_params.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved: {out_csv}")

    out_md = os.path.join(output_dir, "model_best_params.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(df_to_markdown(df_params))
    print(f"[INFO] Saved: {out_md}")

def export_environment_txt(output_dir: str):
    """Export Python version, platform, and key package versions."""
    out_txt = os.path.join(output_dir, "environment.txt")
    lines = [
        "=== Reproducibility environment ===",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Working directory: {os.getcwd()}",
        "",
        f"Python: {sys.version.replace(chr(10), ' ')}",
        f"Platform: {platform.platform()}",
        "",
        "Key package versions:",
    ]

    for pkg in ["numpy", "pandas", "scikit-learn", "scipy",
                "matplotlib", "seaborn", "lifelines", "shap",
                "joblib", "auton_survival"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            lines.append(f"  {pkg}: {ver}")
        except ImportError:
            lines.append(f"  {pkg}: not installed")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[INFO] Saved: {out_txt}")

try:
    export_best_params_table(args.outdir)
    export_environment_txt(args.outdir)
except Exception as e:
    print(f"[WARN] Export of summary tables failed: {e}")

# ============================================================
# 11) Visualisation: time-dependent metrics
# ============================================================

def plot_metric_dot(
    metric_dict: dict,
    ylabel: str,
    title: str,
    ax=None,
):
    """
    Dot plot comparing a scalar metric across models.
    Each model is shown as a coloured dot with its value
    annotated above.
    """
    if ax is None:
        plt.figure(figsize=(7.2, 4.2))
        ax = plt.gca()

    items = [
        (k, float(np.asarray(v).reshape(-1)[0]))
        for k, v in metric_dict.items()
    ]
    dfm = (pd.DataFrame(items, columns=["model", "value"])
             .sort_values("value", ascending=True)
             .reset_index(drop=True))

    colors_map = {
        "CoxPH": COLOR_CPH, "DeepSurv": COLOR_DS,
        "DCM": COLOR_DCM, "RSF": COLOR_RSF,
    }
    colors = [colors_map.get(m, "gray") for m in dfm["model"]]

    vmin, vmax = float(dfm["value"].min()), float(dfm["value"].max())
    pad = max(0.01, 0.35 * (vmax - vmin) if vmax > vmin else 0.02)
    y0, y1 = max(0.0, vmin - pad), vmax + pad

    x = np.arange(len(dfm))
    ax.vlines(x=x, ymin=y0, ymax=dfm["value"].values,
              color="lightgray", linewidth=2.2, zorder=1)
    ax.scatter(x, dfm["value"].values, s=160, c=colors,
               edgecolors="black", linewidth=0.6, zorder=2)

    offset = 0.06 * (y1 - y0)
    for i, v in enumerate(dfm["value"].values):
        ax.text(i, v + offset, f"{v:.3f}",
                ha="center", va="bottom", fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(dfm["model"].values)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model")
    ax.set_title(title)
    ax.set_ylim(y0, y1)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    sns.despine(ax=ax)

# ── 3-panel combined figure (AUC + C-index + IBS) ─────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

# Panel a: Time-dependent AUC
ax = axes[0]
for label, res, color in [
    ("CoxPH",   res_cph,  COLOR_CPH),
    ("DeepSurv", res_dcph, COLOR_DS),
    ("DCM",     res_dcm,  COLOR_DCM),
    ("RSF",     res_rsf,  COLOR_RSF),
]:
    ax.plot(times, np.asarray(res["AUC"]),
            color=color, label=label, lw=1.9)
ax.set_xlabel("Time (years)")
ax.set_ylabel("AUC(t)")
ax.set_title(sci_title("auc_panel_A"))
ax.set_ylim(0, 1.05)
ax.set_xlim(times[0], times[-1] * 1.05)
ax.legend(frameon=False)
sns.despine(ax=ax)

# Panel b: Time-dependent C-index
ax = axes[1]
for label, res, color in [
    ("CoxPH",   res_cph,  COLOR_CPH),
    ("DeepSurv", res_dcph, COLOR_DS),
    ("DCM",     res_dcm,  COLOR_DCM),
    ("RSF",     res_rsf,  COLOR_RSF),
]:
    ax.plot(times, np.asarray(res["Concordance Index"]),
            color=color, label=label, lw=1.9)
ax.set_xlabel("Time (years)")
ax.set_ylabel("C-index(t)")
ax.set_title(sci_title("ctd_panel_B"))
ax.set_ylim(0, 1.05)
ax.set_xlim(times[0], times[-1] * 1.05)
ax.legend(frameon=False)
sns.despine(ax=ax)

# Panel c: IBS dot plot
plot_metric_dot(
    {"CoxPH":   res_cph["IBS"],
     "DeepSurv": res_dcph["IBS"],
     "DCM":     res_dcm["IBS"],
     "RSF":     res_rsf["IBS"]},
    ylabel="IBS",
    title=sci_title("ibs_panel_C"),
    ax=axes[2],
)
plt.tight_layout()
savefig("metrics_combined_auc_cindex_ibs.png", args.outdir)

# ── Standalone AUC figure ──────────────────────────────────
plt.figure(figsize=(6.8, 4.8))
ax = plt.gca()
for label, res, color in [
    ("CoxPH",   res_cph,  COLOR_CPH),
    ("DeepSurv", res_dcph, COLOR_DS),
    ("DCM",     res_dcm,  COLOR_DCM),
    ("RSF",     res_rsf,  COLOR_RSF),
]:
    ax.plot(times, np.asarray(res["AUC"]),
            color=color, label=label, lw=1.9)
ax.set_xlabel("Time (years)")
ax.set_ylabel("AUC(t)")
ax.set_title(sci_title("auc_line"))
ax.set_ylim(0, 1.05)
ax.set_xlim(times[0], times[-1] * 1.05)
ax.legend(frameon=False)
sns.despine(ax=ax)
plt.tight_layout()
savefig("auc_line.png", args.outdir)

# ── Standalone C-index figure ──────────────────────────────
plt.figure(figsize=(6.8, 4.8))
ax = plt.gca()
for label, res, color in [
    ("CoxPH",   res_cph,  COLOR_CPH),
    ("DeepSurv", res_dcph, COLOR_DS),
    ("DCM",     res_dcm,  COLOR_DCM),
    ("RSF",     res_rsf,  COLOR_RSF),
]:
    ax.plot(times, np.asarray(res["Concordance Index"]),
            color=color, label=label, lw=1.9)
ax.set_xlabel("Time (years)")
ax.set_ylabel("C-index(t)")
ax.set_title(sci_title("ctd_line"))
ax.set_ylim(0, 1.05)
ax.set_xlim(times[0], times[-1] * 1.05)
ax.legend(frameon=False)
sns.despine(ax=ax)
plt.tight_layout()
savefig("cindex_line.png", args.outdir)

# ── Standalone IBS dot plot ────────────────────────────────
plt.figure(figsize=(7.2, 4.2))
plot_metric_dot(
    {"CoxPH":   res_cph["IBS"],
     "DeepSurv": res_dcph["IBS"],
     "DCM":     res_dcm["IBS"],
     "RSF":     res_rsf["IBS"]},
    ylabel="IBS",
    title=sci_title("ibs_dot"),
)
plt.tight_layout()
savefig("ibs_dot.png", args.outdir)

# ============================================================
# 11.5) Predicted survival curves on TEST, stratified by
#        TRAIN-derived cut-points (median or IPCW-Youden)
# ============================================================

def mean_ci_curve(
    curves: np.ndarray, alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute pointwise mean and 95% CI from an (n, T) array."""
    curves = np.asarray(curves, dtype=float)
    if curves.ndim != 2:
        raise ValueError(
            f"curves must be 2-D (n, T); got shape={curves.shape}")
    n = curves.shape[0]
    if n <= 1:
        m = curves.reshape(-1)
        return m, m, m
    m  = np.nanmean(curves, axis=0)
    se = np.nanstd(curves, axis=0, ddof=1) / np.sqrt(n)
    z  = 1.96
    lo = np.clip(m - z * se, 0.0, 1.0)
    hi = np.clip(m + z * se, 0.0, 1.0)
    m  = np.clip(m, 0.0, 1.0)
    return m, lo, hi

def risk_at_time_idx(
    model, X: pd.DataFrame, times_list, t_idx: int,
) -> np.ndarray:
    """Return 1 - S(t) at a single time index for every subject."""
    surv = predict_survival(model, X.reset_index(drop=True), times_list)
    return 1.0 - np.asarray(surv[:, t_idx], dtype=float)

def ipcw_youden_cutoff_train(
    time_tr: np.ndarray,
    event_tr: np.ndarray,
    risk_tr: np.ndarray,
    t0: float,
    n_thresholds: int = 200,
) -> float:
    """
    Select an optimal risk cut-point on TRAIN data by maximising
    Youden's J statistic on the IPCW ROC curve at horizon t0.
    """
    time_tr  = np.asarray(time_tr, dtype=float).ravel()
    event_tr = np.asarray(event_tr, dtype=int).ravel()
    risk_tr  = np.asarray(risk_tr, dtype=float).ravel()

    kmc = fit_censoring_km(time_tr, event_tr)
    n_thresholds = max(20, int(n_thresholds))
    thresholds = np.unique(
        np.quantile(risk_tr, np.linspace(0.0, 1.0, n_thresholds)))
    if thresholds.size < 2:
        return float(np.median(risk_tr))

    fpr, tpr, th = ipcw_roc(
        time_tr, event_tr, risk_tr, float(t0), kmc,
        thresholds=thresholds)
    j = tpr - fpr
    if j.size == 0 or not np.isfinite(j).any():
        return float(np.median(risk_tr))
    return float(th[int(np.nanargmax(j))])

def plot_predsurv_panel_test_strat(
    horizon_req: float, cutoff_type: str, output_dir: str,
):
    """
    1×2 panel: predicted survival S(t) curves for low/high risk
    groups on the TEST set, stratified using TRAIN-derived
    cut-points (median or IPCW-Youden).
    """
    cutoff_type = str(cutoff_type).strip().lower()
    if cutoff_type not in ("median", "ipcw_youden"):
        raise ValueError(
            "cutoff_type must be 'median' or 'ipcw_youden'")

    t_idx, t_used = nearest_time_idx(times, float(horizon_req))

    time_tr = (y_tr_model.reset_index(drop=True)["time"]
               .to_numpy(dtype=float))
    event_tr = (y_tr_model.reset_index(drop=True)["event"]
                .to_numpy(dtype=int))

    ds_risk_tr  = risk_at_time_idx(model_dcph, x_tr, times, t_idx)
    rsf_risk_tr = risk_at_time_idx(model_rsf,  x_tr, times, t_idx)

    if cutoff_type == "median":
        ds_cut  = float(np.median(ds_risk_tr))
        rsf_cut = float(np.median(rsf_risk_tr))
    else:
        ds_cut = ipcw_youden_cutoff_train(
            time_tr, event_tr, ds_risk_tr, t0=t_used,
            n_thresholds=IPCW_YOUDEN_N_THRESHOLDS)
        rsf_cut = ipcw_youden_cutoff_train(
            time_tr, event_tr, rsf_risk_tr, t0=t_used,
            n_thresholds=IPCW_YOUDEN_N_THRESHOLDS)

    ds_risk_te  = risk_at_time_idx(model_dcph, x_te, times, t_idx)
    rsf_risk_te = risk_at_time_idx(model_rsf,  x_te, times, t_idx)

    ds_high  = ds_risk_te  >= ds_cut
    rsf_high = rsf_risk_te >= rsf_cut
    ds_low   = ~ds_high
    rsf_low  = ~rsf_high

    ds_surv_te  = predict_survival(
        model_dcph, x_te.reset_index(drop=True), times)
    rsf_surv_te = predict_survival(
        model_rsf,  x_te.reset_index(drop=True), times)

    ds_low_m,  ds_low_lo,  ds_low_hi  = mean_ci_curve(ds_surv_te[ds_low])
    ds_high_m, ds_high_lo, ds_high_hi = mean_ci_curve(ds_surv_te[ds_high])
    rsf_low_m,  rsf_low_lo,  rsf_low_hi  = mean_ci_curve(rsf_surv_te[rsf_low])
    rsf_high_m, rsf_high_lo, rsf_high_hi = mean_ci_curve(rsf_surv_te[rsf_high])

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2), sharey=True)
    fig.suptitle(
        sci_title("predsurv_overall",
                  horizon=float(horizon_req),
                  cutoff_type=cutoff_type),
        y=0.995)

    # Left: DeepSurv
    ax = axes[0]
    ax.step(times, ds_low_m, where="post", color=COLOR_DS, lw=2.0,
            label=f"Low (n={int(ds_low.sum())})")
    ax.fill_between(times, ds_low_lo, ds_low_hi, step="post",
                     color=COLOR_DS, alpha=0.18, linewidth=0)
    ax.step(times, ds_high_m, where="post", color=COLOR_RSF, lw=2.0,
            label=f"High (n={int(ds_high.sum())})")
    ax.fill_between(times, ds_high_lo, ds_high_hi, step="post",
                     color=COLOR_RSF, alpha=0.18, linewidth=0)
    ax.set_title(sci_title("predsurv_A_deepsurv",
                            horizon=float(horizon_req)))
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Predicted survival probability, S(t)")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(float(times[0]), float(times[-1]) * 1.05)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    sns.despine(ax=ax)

    # Right: RSF
    ax = axes[1]
    ax.step(times, rsf_low_m, where="post", color=COLOR_DS, lw=2.0,
            label=f"Low (n={int(rsf_low.sum())})")
    ax.fill_between(times, rsf_low_lo, rsf_low_hi, step="post",
                     color=COLOR_DS, alpha=0.18, linewidth=0)
    ax.step(times, rsf_high_m, where="post", color=COLOR_RSF, lw=2.0,
            label=f"High (n={int(rsf_high.sum())})")
    ax.fill_between(times, rsf_high_lo, rsf_high_hi, step="post",
                     color=COLOR_RSF, alpha=0.18, linewidth=0)
    ax.set_title(sci_title("predsurv_B_rsf",
                            horizon=float(horizon_req)))
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(float(times[0]), float(times[-1]) * 1.05)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    sns.despine(ax=ax)

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    savefig(f"predsurv_panel_test_strat_{cutoff_type}"
            f"_{int(horizon_req)}y_deepsurv_vs_rsf.png",
            output_dir)

    # Export cut-point JSON for external inference
    if EXPORT_ARTIFACTS:
        try:
            cut_info = {
                "horizon_req_years": float(horizon_req),
                "grid_used_years": float(t_used),
                "cutoff_type": cutoff_type,
                "DeepSurv_cutoff": float(ds_cut),
                "RSF_cutoff": float(rsf_cut),
                "IPCW_YOUDEN_N_THRESHOLDS": int(
                    IPCW_YOUDEN_N_THRESHOLDS),
            }
            fn = os.path.join(
                args.artifacts,
                f"cutoffs_train_{cutoff_type}_req{int(horizon_req)}y.json")
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(cut_info, f, ensure_ascii=False, indent=2)
            print(f"[INFO] Saved: {fn}")
        except Exception as e:
            print(f"[WARN] Save cutoffs JSON failed: {e}")

if RUN_DS_RSF_TEST_STRAT_PRED_SURV_PANEL_TRAIN_MEDIAN:
    for h in PRED_SURV_PANEL_HORIZONS_REQ:
        plot_predsurv_panel_test_strat(
            float(h), cutoff_type="median",
            output_dir=args.outdir)

if RUN_DS_RSF_TEST_STRAT_PRED_SURV_PANEL_TRAIN_IPCW_YOUDEN:
    for h in PRED_SURV_PANEL_HORIZONS_REQ:
        plot_predsurv_panel_test_strat(
            float(h), cutoff_type="ipcw_youden",
            output_dir=args.outdir)

# ============================================================
# 12) DeepSurv vs RSF: ROC, calibration, DCA, KM, NB diff
# ============================================================

def plot_overlay_roc():
    """Time-dependent ROC overlay at all evaluation horizons."""
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    kmc_te = fit_censoring_km(
        y_te_model.reset_index(drop=True).time.values,
        y_te_model.reset_index(drop=True).event.values)

    for i, t_req in enumerate(EVAL_HORIZONS_REQ):
        t_idx, t_used = nearest_time_idx(times, t_req)
        surv_ds = predict_survival(
            model_dcph, x_te.reset_index(drop=True), times)[:, t_idx]
        surv_rsf = predict_survival(
            model_rsf, x_te.reset_index(drop=True), times)[:, t_idx]

        fpr_ds, tpr_ds, _ = ipcw_roc(
            y_te_model.time.values, y_te_model.event.values,
            1.0 - surv_ds, t_used, kmc_te)
        fpr_rsf, tpr_rsf, _ = ipcw_roc(
            y_te_model.time.values, y_te_model.event.values,
            1.0 - surv_rsf, t_used, kmc_te)

        auc_ds  = float(auc(fpr_ds, tpr_ds))
        auc_rsf = float(auc(fpr_rsf, tpr_rsf))
        color = SCI_PALETTE[i % len(SCI_PALETTE)]

        ax.step(fpr_ds, tpr_ds, where="post", color=color,
                lw=2.0, linestyle="-",
                label=f"DeepSurv ({_h_label(t_req)}) "
                      f"AUC={auc_ds:.3f}")
        ax.step(fpr_rsf, tpr_rsf, where="post", color=color,
                lw=2.0, linestyle="--",
                label=f"RSF ({_h_label(t_req)}) "
                      f"AUC={auc_rsf:.3f}")

    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(sci_title("roc_overlay"))
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    sns.despine(ax=ax)
    plt.tight_layout()
    savefig("overlay_roc_deepsurv_vs_rsf.png", args.outdir)

def plot_overlay_calibration():
    """Calibration overlay at all evaluation horizons."""
    fig, ax = plt.subplots(figsize=(6.8, 5.2))

    for i, t_req in enumerate(EVAL_HORIZONS_REQ):
        t_idx, t_used = nearest_time_idx(times, t_req)
        surv_ds = predict_survival(
            model_dcph, x_te.reset_index(drop=True), times)[:, t_idx]
        surv_rsf = predict_survival(
            model_rsf, x_te.reset_index(drop=True), times)[:, t_idx]

        for surv, ls, name in [
            (surv_ds, "-", "DeepSurv"),
            (surv_rsf, "--", "RSF"),
        ]:
            bins = np.percentile(surv, [0, 25, 50, 75, 100])
            groups = np.digitize(surv, bins) - 1
            obs_rate, pred_mean = [], []
            for g in range(4):
                mask = groups == g
                if mask.sum() == 0:
                    continue
                kmf = KaplanMeierFitter()
                kmf.fit(y_te_model["time"][mask],
                        y_te_model["event"][mask])
                obs_rate.append(
                    float(min(kmf.predict(t_used), 1.0)))
                pred_mean.append(
                    float(min(surv[mask].mean(), 1.0)))

            color = SCI_PALETTE[i % len(SCI_PALETTE)]
            ax.plot(pred_mean, obs_rate, marker="o", lw=1.8,
                    color=color, linestyle=ls,
                    label=f"{name} ({_h_label(t_req)})")

    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
    ax.set_xlabel("Predicted survival probability")
    ax.set_ylabel("Observed survival probability")
    ax.set_title(sci_title("cal_overlay"))
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    sns.despine(ax=ax)
    plt.tight_layout()
    savefig("overlay_calibration_deepsurv_vs_rsf.png", args.outdir)

def plot_overlay_dca(horizon_req: float):
    """Decision curve analysis at a single horizon."""
    thresholds = np.linspace(0.01, 0.99, 99)
    t_idx, t_used = nearest_time_idx(times, horizon_req)

    surv_ds = predict_survival(
        model_dcph, x_te.reset_index(drop=True), times)[:, t_idx]
    surv_rsf = predict_survival(
        model_rsf, x_te.reset_index(drop=True), times)[:, t_idx]

    time_te  = y_te_model.time.values
    event_te = y_te_model.event.values

    nb_ds   = stdca_ipcw(time_te, event_te, 1.0 - surv_ds,
                          t_used, thresholds)
    nb_rsf  = stdca_ipcw(time_te, event_te, 1.0 - surv_rsf,
                          t_used, thresholds)
    nb_all  = treat_all_nb(time_te, event_te, t_used, thresholds)
    nb_none = treat_none_nb(thresholds)

    plt.figure(figsize=(6.8, 5.2))
    plt.plot(thresholds * 100, nb_ds, label="DeepSurv",
             color=COLOR_DS, lw=2.0)
    plt.plot(thresholds * 100, nb_rsf, label="RSF",
             color=COLOR_RSF, lw=2.0, linestyle="--")
    plt.plot(thresholds * 100, nb_all, label="Treat all",
             color="black", lw=1.2, linestyle="--")
    plt.plot(thresholds * 100, nb_none, label="Treat none",
             color="grey", lw=1.2, linestyle=":")
    plt.xlabel("Threshold probability (%)")
    plt.ylabel("Net benefit")
    plt.title(sci_title("dca", horizon=float(horizon_req)))
    ymax = max(float(np.nanmax(nb_ds)),
               float(np.nanmax(nb_rsf)),
               float(np.nanmax(nb_all)))
    plt.ylim(-0.05, ymax * 1.1 if ymax > 0 else 0.05)
    plt.xlim(0, 100)
    plt.legend(frameon=False)
    sns.despine()
    plt.tight_layout()
    savefig(f"overlay_dca_{int(horizon_req)}yr_deepsurv_vs_rsf.png",
            args.outdir)

def plot_combined_2x2_panel():
    """2×2 panel: ROC, calibration, DCA@5y, DCA@9y."""
    if len(EVAL_HORIZONS_REQ) < 2:
        raise ValueError(
            "EVAL_HORIZONS_REQ must contain >= 2 horizons.")
    h5, h9 = float(EVAL_HORIZONS_REQ[0]), float(EVAL_HORIZONS_REQ[1])
    thresholds = np.linspace(0.01, 0.99, 99)

    fig, axes = plt.subplots(2, 2, figsize=(14.8, 10.8))
    fig.suptitle(sci_title("combined_2x2"), y=0.995)

    time_te  = y_te_model.time.values
    event_te = y_te_model.event.values
    kmc_te   = fit_censoring_km(time_te, event_te)

    # ── a) ROC ─────────────────────────────────────────────
    ax = axes[0, 0]
    for i, t_req in enumerate(EVAL_HORIZONS_REQ):
        t_idx, t_used = nearest_time_idx(times, t_req)
        surv_ds  = predict_survival(
            model_dcph, x_te.reset_index(drop=True), times)[:, t_idx]
        surv_rsf = predict_survival(
            model_rsf, x_te.reset_index(drop=True), times)[:, t_idx]
        fpr_ds, tpr_ds, _ = ipcw_roc(
            time_te, event_te, 1.0 - surv_ds, t_used, kmc_te)
        fpr_rsf, tpr_rsf, _ = ipcw_roc(
            time_te, event_te, 1.0 - surv_rsf, t_used, kmc_te)
        auc_ds  = float(auc(fpr_ds, tpr_ds))
        auc_rsf = float(auc(fpr_rsf, tpr_rsf))
        color = SCI_PALETTE[i % len(SCI_PALETTE)]
        ax.step(fpr_ds, tpr_ds, where="post", color=color,
                lw=2.0, linestyle="-",
                label=f"DeepSurv ({_h_label(t_req)}) "
                      f"AUC={auc_ds:.3f}")
        ax.step(fpr_rsf, tpr_rsf, where="post", color=color,
                lw=2.0, linestyle="--",
                label=f"RSF ({_h_label(t_req)}) "
                      f"AUC={auc_rsf:.3f}")
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", lw=1)
    ax.set_title(sci_title("combined_A"))
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    sns.despine(ax=ax)

    # ── b) Calibration ─────────────────────────────────────
    ax = axes[0, 1]
    for i, t_req in enumerate(EVAL_HORIZONS_REQ):
        t_idx, t_used = nearest_time_idx(times, t_req)
        surv_ds  = predict_survival(
            model_dcph, x_te.reset_index(drop=True), times)[:, t_idx]
        surv_rsf = predict_survival(
            model_rsf, x_te.reset_index(drop=True), times)[:, t_idx]
        for surv, ls, name in [
            (surv_ds, "-", "DeepSurv"),
            (surv_rsf, "--", "RSF"),
        ]:
            bins = np.percentile(surv, [0, 25, 50, 75, 100])
            groups = np.digitize(surv, bins) - 1
            obs_rate, pred_mean = [], []
            for g in range(4):
                mask = groups == g
                if mask.sum() == 0:
                    continue
                kmf = KaplanMeierFitter()
                kmf.fit(y_te_model["time"][mask],
                        y_te_model["event"][mask])
                obs_rate.append(
                    float(min(kmf.predict(t_used), 1.0)))
                pred_mean.append(
                    float(min(surv[mask].mean(), 1.0)))
            color = SCI_PALETTE[i % len(SCI_PALETTE)]
            ax.plot(pred_mean, obs_rate, marker="o", lw=1.8,
                    color=color, linestyle=ls,
                    label=f"{name} ({_h_label(t_req)})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
    ax.set_title(sci_title("combined_B"))
    ax.set_xlabel("Predicted survival probability")
    ax.set_ylabel("Observed survival probability")
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    sns.despine(ax=ax)

    # ── c) DCA at ~5y ──────────────────────────────────────
    ax = axes[1, 0]
    t_idx5, t_used5 = nearest_time_idx(times, h5)
    surv_ds_5  = predict_survival(
        model_dcph, x_te.reset_index(drop=True), times)[:, t_idx5]
    surv_rsf_5 = predict_survival(
        model_rsf, x_te.reset_index(drop=True), times)[:, t_idx5]
    nb_ds_5  = stdca_ipcw(time_te, event_te, 1.0 - surv_ds_5,
                           t_used5, thresholds)
    nb_rsf_5 = stdca_ipcw(time_te, event_te, 1.0 - surv_rsf_5,
                           t_used5, thresholds)
    nb_all_5 = treat_all_nb(time_te, event_te, t_used5, thresholds)
    nb_none  = treat_none_nb(thresholds)
    ax.plot(thresholds * 100, nb_ds_5, label="DeepSurv",
            color=COLOR_DS, lw=2.0)
    ax.plot(thresholds * 100, nb_rsf_5, label="RSF",
            color=COLOR_RSF, lw=2.0, linestyle="--")
    ax.plot(thresholds * 100, nb_all_5, label="Treat all",
            color="black", lw=1.2, linestyle="--")
    ax.plot(thresholds * 100, nb_none, label="Treat none",
            color="grey", lw=1.2, linestyle=":")
    ymax5 = max(float(np.nanmax(nb_ds_5)),
                float(np.nanmax(nb_rsf_5)),
                float(np.nanmax(nb_all_5)))
    ax.set_ylim(-0.05, ymax5 * 1.1 if ymax5 > 0 else 0.05)
    ax.set_title(sci_title("combined_C"))
    ax.set_xlabel("Threshold probability (%)")
    ax.set_ylabel("Net benefit")
    ax.set_xlim(0, 100)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    sns.despine(ax=ax)

    # ── d) DCA at ~9y ──────────────────────────────────────
    ax = axes[1, 1]
    t_idx9, t_used9 = nearest_time_idx(times, h9)
    surv_ds_9  = predict_survival(
        model_dcph, x_te.reset_index(drop=True), times)[:, t_idx9]
    surv_rsf_9 = predict_survival(
        model_rsf, x_te.reset_index(drop=True), times)[:, t_idx9]
    nb_ds_9  = stdca_ipcw(time_te, event_te, 1.0 - surv_ds_9,
                           t_used9, thresholds)
    nb_rsf_9 = stdca_ipcw(time_te, event_te, 1.0 - surv_rsf_9,
                           t_used9, thresholds)
    nb_all_9 = treat_all_nb(time_te, event_te, t_used9, thresholds)
    ax.plot(thresholds * 100, nb_ds_9, label="DeepSurv",
            color=COLOR_DS, lw=2.0)
    ax.plot(thresholds * 100, nb_rsf_9, label="RSF",
            color=COLOR_RSF, lw=2.0, linestyle="--")
    ax.plot(thresholds * 100, nb_all_9, label="Treat all",
            color="black", lw=1.2, linestyle="--")
    ax.plot(thresholds * 100, nb_none, label="Treat none",
            color="grey", lw=1.2, linestyle=":")
    ymax9 = max(float(np.nanmax(nb_ds_9)),
                float(np.nanmax(nb_rsf_9)),
                float(np.nanmax(nb_all_9)))
    ax.set_ylim(-0.05, ymax9 * 1.1 if ymax9 > 0 else 0.05)
    ax.set_title(sci_title("combined_D"))
    ax.set_xlabel("Threshold probability (%)")
    ax.set_ylabel("Net benefit")
    ax.set_xlim(0, 100)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    sns.despine(ax=ax)

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    savefig("overlay_combined_roc_cal_dca5_dca9.png", args.outdir)

def plot_nb_diff(horizon_req: float):
    """Net benefit difference (DeepSurv - RSF) at one horizon."""
    thresholds = np.linspace(0.01, 0.99, 99)
    t_idx, t_used = nearest_time_idx(times, horizon_req)
    surv_ds  = predict_survival(
        model_dcph, x_te.reset_index(drop=True), times)[:, t_idx]
    surv_rsf = predict_survival(
        model_rsf, x_te.reset_index(drop=True), times)[:, t_idx]
    nb_ds  = stdca_ipcw(y_te_model.time.values,
                         y_te_model.event.values,
                         1.0 - surv_ds, t_used, thresholds)
    nb_rsf = stdca_ipcw(y_te_model.time.values,
                         y_te_model.event.values,
                         1.0 - surv_rsf, t_used, thresholds)
    nb_diff = nb_ds - nb_rsf

    plt.figure(figsize=(6.8, 5.2))
    plt.plot(thresholds * 100, nb_diff, color=COLOR_NB, lw=2.2,
             label="DeepSurv \u2212 RSF")
    plt.axhline(0.0, color="black", lw=1.0, linestyle="--")
    plt.xlabel("Threshold probability (%)")
    plt.ylabel("Net benefit difference")
    plt.title(sci_title("nb_diff"))
    plt.xlim(0, 100)
    plt.legend(frameon=False)
    sns.despine()
    plt.tight_layout()
    savefig(f"nb_diff_{int(horizon_req)}yr.png", args.outdir)

def plot_nb_diff_panel():
    """1×2 panel: NB difference at ~5y and ~9y."""
    thresholds = np.linspace(0.01, 0.99, 99)
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2))
    subtitles = [sci_title("nb_diff_panel_A"),
                 sci_title("nb_diff_panel_B")]

    for ax, h, sub in zip(axes, EVAL_HORIZONS_REQ, subtitles):
        t_idx, t_used = nearest_time_idx(times, float(h))
        surv_ds  = predict_survival(
            model_dcph, x_te.reset_index(drop=True), times)[:, t_idx]
        surv_rsf = predict_survival(
            model_rsf, x_te.reset_index(drop=True), times)[:, t_idx]
        nb_ds  = stdca_ipcw(y_te_model.time.values,
                             y_te_model.event.values,
                             1.0 - surv_ds, t_used, thresholds)
        nb_rsf = stdca_ipcw(y_te_model.time.values,
                             y_te_model.event.values,
                             1.0 - surv_rsf, t_used, thresholds)
        nb_diff = nb_ds - nb_rsf
        ax.plot(thresholds * 100, nb_diff, color=COLOR_NB, lw=2.2,
                label="DeepSurv \u2212 RSF")
        ax.axhline(0.0, color="black", lw=1.0, linestyle="--")
        ax.set_title(sub)
        ax.set_xlabel("Threshold probability (%)")
        ax.set_ylabel("Net benefit difference")
        ax.set_xlim(0, 100)
        ax.legend(frameon=False, fontsize=9, loc="upper right")
        sns.despine(ax=ax)

    plt.tight_layout()
    savefig("nb_diff_5yr_9yr_panel.png", args.outdir)

def plot_km_panel(
    horizon_req: float, cutoff_type: str = "median",
):
    """
    1×2 KM panel: DeepSurv vs RSF risk stratification on the
    test set, using training-derived cut-points.
    """
    cutoff_type = str(cutoff_type).strip().lower()
    t_idx, t_used = nearest_time_idx(times, horizon_req)
    time_tr  = (y_tr_model.reset_index(drop=True)["time"]
                .to_numpy(dtype=float))
    event_tr = (y_tr_model.reset_index(drop=True)["event"]
                .to_numpy(dtype=int))

    def _cutoff(model):
        r = risk_at_time_idx(model, x_tr, times, t_idx)
        if cutoff_type == "median":
            return float(np.median(r))
        return ipcw_youden_cutoff_train(
            time_tr, event_tr, r, t0=t_used,
            n_thresholds=IPCW_YOUDEN_N_THRESHOLDS)

    def _groups(model):
        cut = _cutoff(model)
        r_te = risk_at_time_idx(model, x_te, times, t_idx)
        high = r_te >= cut
        low = ~high
        lr = logrank_test(
            y_te_model.time.values[low],
            y_te_model.time.values[high],
            event_observed_A=y_te_model.event.values[low],
            event_observed_B=y_te_model.event.values[high])
        return cut, low, high, float(lr.p_value)

    def _km_step_ci(dur, ev):
        kmf = KaplanMeierFitter()
        kmf.fit(durations=dur, event_observed=ev)
        sf = kmf.survival_function_
        x = sf.index.values.astype(float)
        y = sf.iloc[:, 0].values.astype(float)
        ci = getattr(kmf, "confidence_interval_", None)
        if ci is None or ci.shape[1] < 2:
            return x, y, None, None
        return (x, y,
                ci.iloc[:, 0].values.astype(float),
                ci.iloc[:, 1].values.astype(float))

    def _draw_km(ax, dur, ev, label, color):
        x, y, lo, hi = _km_step_ci(dur, ev)
        ax.step(x, y, where="post", color=color, lw=2.0,
                label=label)
        if lo is not None:
            ax.fill_between(x, lo, hi, step="post",
                             color=color, alpha=0.18, linewidth=0)

    _, low_ds, high_ds, _ = _groups(model_dcph)
    _, low_rsf, high_rsf, _ = _groups(model_rsf)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2))
    fig.suptitle(sci_title("km_overall"), y=0.995)

    ax = axes[0]
    _draw_km(ax, y_te_model.time.values[low_ds],
             y_te_model.event.values[low_ds],
             f"Low (n={int(low_ds.sum())})", COLOR_DS)
    _draw_km(ax, y_te_model.time.values[high_ds],
             y_te_model.event.values[high_ds],
             f"High (n={int(high_ds.sum())})", COLOR_RSF)
    ax.set_title(sci_title("km_A_deepsurv"))
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    sns.despine(ax=ax)

    ax = axes[1]
    _draw_km(ax, y_te_model.time.values[low_rsf],
             y_te_model.event.values[low_rsf],
             f"Low (n={int(low_rsf.sum())})", COLOR_DS)
    _draw_km(ax, y_te_model.time.values[high_rsf],
             y_te_model.event.values[high_rsf],
             f"High (n={int(high_rsf.sum())})", COLOR_RSF)
    ax.set_title(sci_title("km_B_rsf"))
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    sns.despine(ax=ax)

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    savefig(f"km_panel_{int(horizon_req)}yr_{cutoff_type}.png",
            args.outdir)

# ── Run overlay / panel blocks ─────────────────────────────
if RUN_DS_VS_RSF_OVERLAYS:
    plot_overlay_roc()
    plot_overlay_calibration()
    for h in EVAL_HORIZONS_REQ:
        plot_overlay_dca(float(h))
    plot_combined_2x2_panel()

if RUN_DS_VS_RSF_NB_DIFF:
    for h in EVAL_HORIZONS_REQ:
        plot_nb_diff(float(h))
    plot_nb_diff_panel()

if RUN_DS_VS_RSF_KM_PANEL:
    for h in EVAL_HORIZONS_REQ:
        plot_km_panel(float(h), cutoff_type="median")
        plot_km_panel(float(h), cutoff_type="ipcw_youden")

# ============================================================
# 13) Bootstrap: DeepSurv vs RSF on TEST set
# ============================================================

def bootstrap_ci(
    arr: np.ndarray, alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Percentile bootstrap confidence interval."""
    mean = np.nanmean(arr, axis=0)
    low  = np.nanquantile(arr, alpha / 2.0, axis=0)
    high = np.nanquantile(arr, 1.0 - alpha / 2.0, axis=0)
    return mean, low, high

def bootstrap_pvalue(delta: np.ndarray) -> float:
    """Two-sided bootstrap p-value for H0: delta = 0."""
    delta = np.asarray(delta, dtype=float).ravel()
    p_le0 = float(np.mean(delta <= 0))
    p_ge0 = float(np.mean(delta >= 0))
    return float(2.0 * min(p_le0, p_ge0))

def plot_auc_ctd_ci_panel(
    times, auc_ds_mean, auc_ds_lo, auc_ds_hi,
    auc_rsf_mean, auc_rsf_lo, auc_rsf_hi,
    ctd_ds_mean, ctd_ds_lo, ctd_ds_hi,
    ctd_rsf_mean, ctd_rsf_lo, ctd_rsf_hi,
):
    """1×2 panel: AUC(t) and C-index(t) with bootstrap CI."""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8))
    fig.suptitle(sci_title("boot_2panel"), y=0.995)

    ax = axes[0]
    ax.plot(times, auc_ds_mean, color=COLOR_DS, lw=2.2,
            label="DeepSurv")
    ax.fill_between(times, auc_ds_lo, auc_ds_hi,
                     color=COLOR_DS, alpha=0.18, linewidth=0)
    ax.plot(times, auc_rsf_mean, color=COLOR_RSF, lw=2.2,
            linestyle="--", label="RSF")
    ax.fill_between(times, auc_rsf_lo, auc_rsf_hi,
                     color=COLOR_RSF, alpha=0.18, linewidth=0)
    ax.set_xlabel("Time (years)"); ax.set_ylabel("AUC(t)")
    ax.set_title(sci_title("boot_2panel_A"))
    ax.set_ylim(0, 1.05)
    ax.set_xlim(times[0], times[-1] * 1.05)
    ax.legend(frameon=False); sns.despine(ax=ax)

    ax = axes[1]
    ax.plot(times, ctd_ds_mean, color=COLOR_DS, lw=2.2,
            label="DeepSurv")
    ax.fill_between(times, ctd_ds_lo, ctd_ds_hi,
                     color=COLOR_DS, alpha=0.18, linewidth=0)
    ax.plot(times, ctd_rsf_mean, color=COLOR_RSF, lw=2.2,
            linestyle="--", label="RSF")
    ax.fill_between(times, ctd_rsf_lo, ctd_rsf_hi,
                     color=COLOR_RSF, alpha=0.18, linewidth=0)
    ax.set_xlabel("Time (years)"); ax.set_ylabel("C-index(t)")
    ax.set_title(sci_title("boot_2panel_B"))
    ax.set_ylim(0, 1.05)
    ax.set_xlim(times[0], times[-1] * 1.05)
    ax.legend(frameon=False); sns.despine(ax=ax)

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    savefig("bootstrap_auc_ctd_ci_2panel.png", args.outdir)

def run_bootstrap():
    """
    Stratified bootstrap on the test set: compare DeepSurv vs RSF
    on AUC(t), C-index(t), and IBS with 95% CIs and p-values.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    t_idx_5, t_used_5 = nearest_time_idx(times, 5.0)
    t_idx_9, t_used_9 = nearest_time_idx(times, 9.0)

    te_event = np.asarray(
        y_te_model.reset_index(drop=True)["event"].values,
        dtype=int).ravel()
    idx_event = np.where(te_event == 1)[0]
    idx_cens  = np.where(te_event == 0)[0]
    if idx_event.size == 0:
        raise ValueError("No events in test set.")

    n_event, n_cens = int(idx_event.size), int(idx_cens.size)

    auc_ds_all, auc_rsf_all = [], []
    ctd_ds_all, ctd_rsf_all = [], []
    ibs_ds_all, ibs_rsf_all = [], []
    auc5_ds, auc5_rsf, auc9_ds, auc9_rsf = [], [], [], []
    ctd5_ds, ctd5_rsf, ctd9_ds, ctd9_rsf = [], [], [], []

    print(f"\n[INFO] Bootstrap: B={BOOTSTRAP_B}, "
          f"alpha={BOOTSTRAP_ALPHA}")
    print(f"[INFO] t\u22485y grid={t_used_5:.4f}, "
          f"t\u22489y grid={t_used_9:.4f}")
    print(f"[INFO] Test events={n_event}, censored={n_cens}")

    y_te0 = y_te_model.reset_index(drop=True)
    x_te0 = x_te.reset_index(drop=True)
    y_tr0 = y_tr_model.reset_index(drop=True)

    done, tries = 0, 0
    max_tries = int(BOOTSTRAP_B) * 3

    while done < int(BOOTSTRAP_B) and tries < max_tries:
        tries += 1
        if n_cens > 0:
            idx = np.concatenate([
                rng.choice(idx_event, size=n_event, replace=True),
                rng.choice(idx_cens, size=n_cens, replace=True)])
        else:
            idx = rng.choice(idx_event, size=n_event, replace=True)
        rng.shuffle(idx)

        x_b = x_te0.iloc[idx].reset_index(drop=True)
        y_b = y_te0.iloc[idx].reset_index(drop=True)
        if int(np.sum(y_b["event"].values)) == 0:
            continue

        try:
            pred_ds  = predict_survival(model_dcph, x_b, times)
            pred_rsf = predict_survival(model_rsf, x_b, times)

            a_ds = np.asarray(survival_regression_metric(
                "auc", y_b, pred_ds, times, y_tr0),
                dtype=float).ravel()
            a_rsf = np.asarray(survival_regression_metric(
                "auc", y_b, pred_rsf, times, y_tr0),
                dtype=float).ravel()
            c_ds = np.asarray(survival_regression_metric(
                "ctd", y_b, pred_ds, times, y_tr0),
                dtype=float).ravel()
            c_rsf = np.asarray(survival_regression_metric(
                "ctd", y_b, pred_rsf, times, y_tr0),
                dtype=float).ravel()
            i_ds = float(np.asarray(survival_regression_metric(
                "ibs", y_b, pred_ds, times, y_tr0)).ravel()[0])
            i_rsf = float(np.asarray(survival_regression_metric(
                "ibs", y_b, pred_rsf, times, y_tr0)).ravel()[0])
        except ValueError:
            continue

        auc_ds_all.append(a_ds);  auc_rsf_all.append(a_rsf)
        ctd_ds_all.append(c_ds);  ctd_rsf_all.append(c_rsf)
        ibs_ds_all.append(i_ds);  ibs_rsf_all.append(i_rsf)

        auc5_ds.append(float(a_ds[t_idx_5]))
        auc5_rsf.append(float(a_rsf[t_idx_5]))
        auc9_ds.append(float(a_ds[t_idx_9]))
        auc9_rsf.append(float(a_rsf[t_idx_9]))
        ctd5_ds.append(float(c_ds[t_idx_5]))
        ctd5_rsf.append(float(c_rsf[t_idx_5]))
        ctd9_ds.append(float(c_ds[t_idx_9]))
        ctd9_rsf.append(float(c_rsf[t_idx_9]))

        done += 1
        if done % 100 == 0:
            print(f"  bootstrap {done}/{BOOTSTRAP_B} "
                  f"(tries={tries})")

    effective_B = int(done)
    if effective_B < int(BOOTSTRAP_B):
        print(f"[WARN] Effective B={effective_B} < "
              f"requested={BOOTSTRAP_B}")

    auc_ds_all  = np.asarray(auc_ds_all, dtype=float)
    auc_rsf_all = np.asarray(auc_rsf_all, dtype=float)
    ctd_ds_all  = np.asarray(ctd_ds_all, dtype=float)
    ctd_rsf_all = np.asarray(ctd_rsf_all, dtype=float)

    auc_ds_m, auc_ds_lo, auc_ds_hi = bootstrap_ci(
        auc_ds_all, BOOTSTRAP_ALPHA)
    auc_rsf_m, auc_rsf_lo, auc_rsf_hi = bootstrap_ci(
        auc_rsf_all, BOOTSTRAP_ALPHA)
    ctd_ds_m, ctd_ds_lo, ctd_ds_hi = bootstrap_ci(
        ctd_ds_all, BOOTSTRAP_ALPHA)
    ctd_rsf_m, ctd_rsf_lo, ctd_rsf_hi = bootstrap_ci(
        ctd_rsf_all, BOOTSTRAP_ALPHA)

    # Export curve CSVs
    for metric, ds_data, rsf_data, fname in [
        ("auc", (auc_ds_m, auc_ds_lo, auc_ds_hi),
                (auc_rsf_m, auc_rsf_lo, auc_rsf_hi),
                "bootstrap_curve_auc.csv"),
        ("ctd", (ctd_ds_m, ctd_ds_lo, ctd_ds_hi),
                (ctd_rsf_m, ctd_rsf_lo, ctd_rsf_hi),
                "bootstrap_curve_ctd.csv"),
    ]:
        df_curve = pd.DataFrame({
            "time_years": times,
            "deepsurv_mean": ds_data[0],
            "deepsurv_ci_low": ds_data[1],
            "deepsurv_ci_high": ds_data[2],
            "rsf_mean": rsf_data[0],
            "rsf_ci_low": rsf_data[1],
            "rsf_ci_high": rsf_data[2],
        })
        out = os.path.join(args.outdir, fname)
        df_curve.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[INFO] Saved: {out}")

    # Horizon-specific summary with p-values
    def _summ(name, ds_vals, rsf_vals):
        ds_vals  = np.asarray(ds_vals, dtype=float).ravel()
        rsf_vals = np.asarray(rsf_vals, dtype=float).ravel()
        delta = ds_vals - rsf_vals
        m_d, lo_d, hi_d = bootstrap_ci(delta[:, None],
                                        BOOTSTRAP_ALPHA)
        m_ds, lo_ds, hi_ds = bootstrap_ci(ds_vals[:, None],
                                           BOOTSTRAP_ALPHA)
        m_r, lo_r, hi_r = bootstrap_ci(rsf_vals[:, None],
                                        BOOTSTRAP_ALPHA)
        p = bootstrap_pvalue(delta)
        return {
            "metric": name,
            "effective_B": effective_B,
            "deepsurv_mean": float(m_ds[0]),
            "deepsurv_ci_low": float(lo_ds[0]),
            "deepsurv_ci_high": float(hi_ds[0]),
            "rsf_mean": float(m_r[0]),
            "rsf_ci_low": float(lo_r[0]),
            "rsf_ci_high": float(hi_r[0]),
            "delta_mean": float(m_d[0]),
            "delta_ci_low": float(lo_d[0]),
            "delta_ci_high": float(hi_d[0]),
            "p_value": float(p),
        }

    rows = [
        _summ(f"AUC@{_h_label(5.0)}", auc5_ds, auc5_rsf),
        _summ(f"AUC@{_h_label(9.0)}", auc9_ds, auc9_rsf),
        _summ(f"C-index@{_h_label(5.0)}", ctd5_ds, ctd5_rsf),
        _summ(f"C-index@{_h_label(9.0)}", ctd9_ds, ctd9_rsf),
        _summ("IBS", ibs_ds_all, ibs_rsf_all),
    ]
    df_sum = pd.DataFrame(rows)
    out_csv = os.path.join(args.outdir, "bootstrap_summary.csv")
    df_sum.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved: {out_csv}")

    # Standalone AUC CI plot
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(times, auc_ds_m, color=COLOR_DS, lw=2.2,
             label="DeepSurv")
    plt.fill_between(times, auc_ds_lo, auc_ds_hi,
                      color=COLOR_DS, alpha=0.18, linewidth=0)
    plt.plot(times, auc_rsf_m, color=COLOR_RSF, lw=2.2,
             linestyle="--", label="RSF")
    plt.fill_between(times, auc_rsf_lo, auc_rsf_hi,
                      color=COLOR_RSF, alpha=0.18, linewidth=0)
    plt.xlabel("Time (years)"); plt.ylabel("AUC(t)")
    plt.title(sci_title("boot_auc"))
    plt.ylim(0, 1.05); plt.xlim(times[0], times[-1] * 1.05)
    plt.legend(frameon=False); sns.despine()
    plt.tight_layout()
    savefig("bootstrap_auc_ci.png", args.outdir)

    # Standalone C-index CI plot
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(times, ctd_ds_m, color=COLOR_DS, lw=2.2,
             label="DeepSurv")
    plt.fill_between(times, ctd_ds_lo, ctd_ds_hi,
                      color=COLOR_DS, alpha=0.18, linewidth=0)
    plt.plot(times, ctd_rsf_m, color=COLOR_RSF, lw=2.2,
             linestyle="--", label="RSF")
    plt.fill_between(times, ctd_rsf_lo, ctd_rsf_hi,
                      color=COLOR_RSF, alpha=0.18, linewidth=0)
    plt.xlabel("Time (years)"); plt.ylabel("C-index(t)")
    plt.title(sci_title("boot_ctd"))
    plt.ylim(0, 1.05); plt.xlim(times[0], times[-1] * 1.05)
    plt.legend(frameon=False); sns.despine()
    plt.tight_layout()
    savefig("bootstrap_ctd_ci.png", args.outdir)

    # Combined 2-panel
    plot_auc_ctd_ci_panel(
        times,
        auc_ds_m, auc_ds_lo, auc_ds_hi,
        auc_rsf_m, auc_rsf_lo, auc_rsf_hi,
        ctd_ds_m, ctd_ds_lo, ctd_ds_hi,
        ctd_rsf_m, ctd_rsf_lo, ctd_rsf_hi)

if RUN_BOOTSTRAP:
    run_bootstrap()

# ============================================================
# 14) RSF permutation importance (scikit-survival)
# ============================================================
permimp_png_by_horizon: Dict[int, str] = {}

if RUN_RSF_PERM_IMPORTANCE:
    try:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv

        y_tr_struct = Surv.from_arrays(
            event=(y_tr_model.reset_index(drop=True)["event"]
                   .astype(bool).values),
            time=(y_tr_model.reset_index(drop=True)["time"]
                  .astype(float).values))

        rsf_param = res_rsf["best_param"]
        rsf_native = RandomSurvivalForest(
            n_estimators=int(rsf_param.get("n_estimators", 200)),
            max_depth=rsf_param.get("max_depth", None),
            min_samples_split=int(
                rsf_param.get("min_samples_split", 2)),
            random_state=args.seed, n_jobs=-1)

        x_tr0 = x_tr.reset_index(drop=True).copy()
        x_tr0[:] = x_tr0.to_numpy(dtype=float)
        rsf_native.fit(x_tr0, y_tr_struct)

        y_te0 = y_te_model.reset_index(drop=True)
        x_te0 = x_te.reset_index(drop=True).copy()
        x_te0[:] = x_te0.to_numpy(dtype=float)

        feat_raw  = list(x_te0.columns)
        feat_disp = [FEATURE_DISPLAY_NAME.get(str(f), str(f))
                     for f in feat_raw]

        for h_req in PERM_IMPORTANCE_HORIZONS_REQ:
            _, t_used_pi = nearest_time_idx(times, float(h_req))

            def _score_fn(estimator, X, y=None):
                """IPCW-AUC scorer for permutation importance."""
                surv_fns = estimator.predict_survival_function(
                    X, return_array=False)
                risk = np.asarray(
                    [1.0 - fn(t_used_pi) for fn in surv_fns],
                    dtype=float)
                time_te = y_te0["time"].values.astype(float)
                event_te = y_te0["event"].values.astype(int)
                y_bin = ((time_te <= t_used_pi)
                         & (event_te == 1)).astype(int)
                kmc = fit_censoring_km(time_te, event_te)
                w = np.asarray([
                    1.0 / (float(kmc.predict(
                        min(t, t_used_pi))) + 1e-12)
                    for t in time_te], dtype=float)
                w = np.clip(w, 0, np.nanquantile(w, 0.99))
                thresholds = np.linspace(
                    float(np.min(risk)), float(np.max(risk)), 300)
                tpr_l, fpr_l = [], []
                for c in thresholds:
                    pred = (risk >= c).astype(int)
                    tp = np.sum(w * (pred == 1) * (y_bin == 1))
                    fp = np.sum(w * (pred == 1) * (y_bin == 0))
                    fn_ = np.sum(w * (pred == 0) * (y_bin == 1))
                    tn = np.sum(w * (pred == 0) * (y_bin == 0))
                    tpr_l.append(tp / (tp + fn_ + 1e-12))
                    fpr_l.append(fp / (fp + tn + 1e-12))
                return float(auc(np.asarray(fpr_l),
                                 np.asarray(tpr_l)))

            print(f"\n[INFO] RSF permutation importance at "
                  f"{_h_label(h_req)} (grid={t_used_pi:.3f})")

            r = permutation_importance(
                rsf_native, x_te0, y=None,
                scoring=_score_fn,
                n_repeats=int(PERM_IMPORTANCE_N_REPEATS),
                random_state=int(PERM_IMPORTANCE_RANDOM_SEED),
                n_jobs=-1)

            df_pi = (pd.DataFrame({
                "feature_raw": feat_raw,
                "feature": feat_disp,
                "importance_mean": r.importances_mean,
                "importance_std": r.importances_std,
            }).sort_values("importance_mean", ascending=False)
              .reset_index(drop=True))

            out_csv = os.path.join(
                args.outdir,
                f"rsf_perm_importance_{int(h_req)}yr.csv")
            df_pi.to_csv(out_csv, index=False,
                          encoding="utf-8-sig")
            print(f"[INFO] Saved: {out_csv}")

            # Bar plot
            topk = df_pi.head(15).iloc[::-1].reset_index(drop=True)
            vals = topk["importance_mean"].to_numpy(dtype=float)
            vmin, vmax = float(vals.min()), float(vals.max())
            denom = (vmax - vmin) if vmax > vmin else 1.0
            colors = [SHAP_CMAP((v - vmin) / denom) for v in vals]

            plt.figure(figsize=(8.4, 6.2))
            plt.barh(
                np.arange(len(topk)),
                topk["importance_mean"].values,
                xerr=topk["importance_std"].values,
                color=colors, edgecolor="none", alpha=0.95,
                error_kw={"ecolor": "black", "elinewidth": 1.1,
                           "capsize": 2})
            plt.yticks(np.arange(len(topk)),
                        topk["feature"].values)
            plt.xlabel("Permutation importance "
                       "(\u0394 IPCW-AUC)")
            plt.grid(axis="x", linestyle="--", alpha=0.25)
            sns.despine(); plt.tight_layout()

            png = os.path.join(
                args.outdir,
                f"rsf_perm_importance_{int(h_req)}yr.png")
            plt.savefig(png, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[INFO] Saved: {png}")
            permimp_png_by_horizon[int(h_req)] = png

    except Exception as e:
        print(f"[WARN] RSF permutation importance skipped: {e}")

if (EXPORT_PERMIMP_5Y_9Y_PANEL
        and 5 in permimp_png_by_horizon
        and 9 in permimp_png_by_horizon):
    panel_from_two_images(
        permimp_png_by_horizon[5],
        permimp_png_by_horizon[9],
        os.path.join(args.outdir,
                     "rsf_perm_importance_5y_9y_panel.png"),
        sci_title("perm_panel"),
        subtitles=[
            "Permutation importance at t\u22485y",
            "Permutation importance at t\u22489y"])

# ============================================================
# 15) DeepSurv SHAP (Kernel SHAP)
# ============================================================
import re
from matplotlib import gridspec
from scipy.stats import gaussian_kde

def apply_display_names(df_: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to academic display names."""
    out = df_.copy()
    out.columns = [
        FEATURE_DISPLAY_NAME.get(c, re.sub(r"_\d+$", "", str(c)))
        for c in out.columns]
    return out

def shap_combo_plot(
    shap_values: pd.DataFrame,
    X_test: pd.DataFrame,
    outpath: str,
    fig_title: Optional[str] = None,
    dep_plots: int = 10,
    cmap=None,
):
    """
    Custom SHAP beeswarm + dependence panel.
    Both DataFrames must share the same column names.
    """
    set_journal_style()
    if cmap is None:
        cmap = SHAP_CMAP

    shap_df = pd.DataFrame(shap_values.values,
                            columns=X_test.columns)
    dep_plots = int(dep_plots)
    dep_rows = int(np.ceil(dep_plots / 2))

    fig_h = max(15, 3.0 * dep_rows)
    fig = plt.figure(figsize=(22, fig_h), facecolor="white")
    gs = gridspec.GridSpec(dep_rows, 4, figure=fig,
                            wspace=0.45, hspace=0.35)

    # ── Left: beeswarm-like summary ────────────────────────
    ax_main = fig.add_subplot(gs[:, :2])
    mean_abs = np.abs(shap_df).mean(axis=0)
    imp_df = (pd.DataFrame({
        "feature": X_test.columns,
        "importance": mean_abs})
        .sort_values("importance", ascending=True))

    ax_main.set_yticks(range(len(imp_df)))
    ax_main.set_yticklabels(imp_df["feature"], fontsize=14,
                             fontfamily="Arial")

    ax_top = ax_main.twiny()
    ax_top.barh(range(len(imp_df)), imp_df["importance"],
                color="lightgray", alpha=0.5, height=0.7)
    ax_top.set_xlabel("Mean(|SHAP|)", fontsize=14,
                       fontfamily="Arial")
    ax_top.tick_params(axis="x", labelsize=12)
    ax_top.grid(False)

    for i, feat in enumerate(imp_df["feature"]):
        orig_idx = X_test.columns.get_loc(feat)
        sv = shap_df.values[:, orig_idx]
        fv = X_test.iloc[:, orig_idx]
        jitter = np.random.normal(0, 0.08, sv.shape[0])
        ax_main.scatter(sv, i + jitter, c=fv, cmap=cmap,
                         s=15, alpha=1, zorder=2)

    ax_main.set_xlabel("SHAP value", fontsize=14,
                        fontfamily="Arial")
    ax_main.tick_params(axis="x", labelsize=12)
    ax_main.grid(True, axis="x", linestyle="--", alpha=0.6)

    # Colorbar
    fig.canvas.draw()
    pos = ax_main.get_position()
    cax = fig.add_axes([pos.x1 + 0.01, pos.y0,
                         0.015, pos.height])
    norm = plt.Normalize(
        vmin=float(np.nanmin(X_test.values)),
        vmax=float(np.nanmax(X_test.values)))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Feature value", rotation=90, labelpad=-15,
                    fontsize=11, fontfamily="Arial")
    cbar.outline.set_visible(False); cbar.set_ticks([])
    cbar.ax.text(0.6, 1.02, "High", ha="center", va="top",
                  transform=cbar.ax.transAxes, fontsize=12,
                  fontfamily="Arial")
    cbar.ax.text(0.6, -0.02, "Low", ha="center", va="bottom",
                  transform=cbar.ax.transAxes, fontsize=12,
                  fontfamily="Arial")

    # ── Right: dependence plots ────────────────────────────
    topN = min(dep_plots, len(imp_df))
    top_feats = (imp_df["feature"].tail(topN)
                 .iloc[::-1].tolist())

    axes_sc = []
    for r in range(dep_rows):
        for c in range(2):
            axes_sc.append(fig.add_subplot(gs[r, c + 2]))

    for i, feat in enumerate(top_feats):
        ax = axes_sc[i]
        xd = X_test[feat].values
        yd = shap_df[feat].values
        xy = np.vstack([xd, yd])
        z = gaussian_kde(xy)(xy)
        ax.scatter(xd, yd, c=z, s=18, alpha=0.75, cmap=cmap)
        sns.regplot(x=X_test[feat], y=shap_df[feat],
                     scatter=False, lowess=True,
                     color=cmap(0.6), ax=ax)
        ax.axhline(y=0, color="black", linestyle="-.",
                    linewidth=1)
        ax.set_xlabel(feat, fontsize=11, fontfamily="Arial")
        ax.set_ylabel("SHAP main effect", fontsize=11,
                        fontfamily="Arial")
        ax.tick_params(labelsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        force_axes_font(ax, "Arial")

    for j in range(len(top_feats), len(axes_sc)):
        axes_sc[j].axis("off")

    force_axes_font(ax_main, "Arial")
    force_axes_font(ax_top, "Arial")

    if fig_title:
        fig.suptitle(fig_title, fontsize=SHAP_SUPTITLE_FS,
                      fontweight=SHAP_SUPTITLE_FW, y=0.995,
                      fontfamily="Arial")
        plt.tight_layout(rect=[0, 0, 1, 0.985])
    else:
        plt.tight_layout()

    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight",
                 facecolor="white")
    plt.close(fig)
    print(f"[INFO] Saved: {outpath}")

shap_png_by_horizon: Dict[int, str] = {}

if RUN_SHAP:
    try:
        import shap

        x_tr0 = x_tr.reset_index(drop=True).copy()
        x_te0 = x_te.reset_index(drop=True).copy()
        x_tr0[:] = x_tr0.to_numpy(dtype=float)
        x_te0[:] = x_te0.to_numpy(dtype=float)

        bg_n = int(min(SHAP_BG_N, len(x_tr0)))
        x_bg = (x_tr0.sample(bg_n, random_state=42)
                .reset_index(drop=True))
        x_explain = x_te0.reset_index(drop=True)
        np.random.seed(0)

        for h_req in SHAP_HORIZONS_REQ:
            t_idx, t_used = nearest_time_idx(times, h_req)
            print(f"\n[INFO] SHAP (KernelExplainer) at "
                  f"{_h_label(h_req)} (grid={t_used:.4f})")

            def model_risk_fn(X_np):
                X_df = pd.DataFrame(X_np, columns=x_tr0.columns)
                surv = model_dcph.predict_survival(X_df, times)
                return ((1.0 - np.asarray(surv)[:, t_idx])
                        .reshape(-1, 1))

            explainer = shap.KernelExplainer(
                model_risk_fn, x_bg, link="identity")

            n = len(x_explain)
            batch = int(max(1, SHAP_BATCH_SIZE))
            chunks = int(np.ceil(n / batch))
            shap_chunks = []
            for k in range(chunks):
                s, e = k * batch, min(n, (k + 1) * batch)
                raw = explainer.shap_values(
                    x_explain.iloc[s:e],
                    nsamples=int(SHAP_NSAMPLES))
                if isinstance(raw, list):
                    raw = raw[0]
                raw = np.asarray(raw)
                if raw.ndim == 3 and raw.shape[-1] == 1:
                    raw = raw[:, :, 0]
                shap_chunks.append(raw)

            raw_shap = np.vstack(shap_chunks)

            X_disp = apply_display_names(x_explain)
            shap_disp = pd.DataFrame(
                raw_shap, columns=x_tr0.columns)
            shap_disp.columns = X_disp.columns

            out_png = os.path.join(
                args.outdir,
                f"shap_combo_{int(h_req)}yr.png")
            shap_combo_plot(
                shap_values=shap_disp, X_test=X_disp,
                outpath=out_png, fig_title=None,
                dep_plots=SHAP_DEP_PLOTS, cmap=SHAP_CMAP)
            shap_png_by_horizon[int(h_req)] = out_png

            # Export mean |SHAP| ranking
            mean_abs = np.abs(shap_disp.values).mean(axis=0)
            df_rank = (pd.DataFrame({
                "feature": shap_disp.columns,
                "mean_abs_shap": mean_abs})
                .sort_values("mean_abs_shap", ascending=False))
            rank_csv = os.path.join(
                args.outdir,
                f"shap_ranking_{int(h_req)}yr.csv")
            df_rank.to_csv(rank_csv, index=False,
                            encoding="utf-8-sig")
            print(f"[INFO] Saved: {rank_csv}")

    except Exception as e:
        print(f"[WARN] SHAP failed: {e}")

if (EXPORT_SHAP_5Y_9Y_PANEL
        and 5 in shap_png_by_horizon
        and 9 in shap_png_by_horizon):
    panel_from_two_images(
        shap_png_by_horizon[5], shap_png_by_horizon[9],
        os.path.join(args.outdir, "shap_5y_9y_panel.png"),
        sci_title("shap_panel"),
        subtitles=[
            "DeepSurv SHAP at t\u22485y",
            "DeepSurv SHAP at t\u22489y"])

# ============================================================
# 16) Single KM plots + 6-panel composite figures
# ============================================================

def auto_crop_border(img, tol=0.02, pad=4):
    """Crop nearly-uniform border based on corner colours."""
    arr = img.astype(np.float32)
    if arr.max() > 1.5:
        arr /= 255.0
    h, w = arr.shape[:2]
    corners = np.array([arr[0, 0, :3], arr[0, w-1, :3],
                         arr[h-1, 0, :3], arr[h-1, w-1, :3]])
    bg = corners.mean(axis=0)
    diff = np.sqrt(((arr[..., :3] - bg)**2).sum(axis=2))
    mask = diff > tol
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return img
    y0 = max(0, ys.min() - pad)
    y1 = min(h, ys.max() + pad + 1)
    x0 = max(0, xs.min() - pad)
    x1 = min(w, xs.max() + pad + 1)
    return img[y0:y1, x0:x1, ...]

def plot_km_single(
    model, model_label: str, horizon_req: float,
    cutoff_type: str, out_png: str,
):
    """Generate a single KM plot for one model and horizon."""
    cutoff_type = str(cutoff_type).strip().lower()
    t_idx, t_used = nearest_time_idx(times, float(horizon_req))
    r_tr = risk_at_time_idx(model, x_tr, times, t_idx)

    if cutoff_type == "median":
        cut = float(np.median(r_tr))
    else:
        time_tr = (y_tr_model.reset_index(drop=True)["time"]
                   .to_numpy(dtype=float))
        event_tr = (y_tr_model.reset_index(drop=True)["event"]
                    .to_numpy(dtype=int))
        cut = ipcw_youden_cutoff_train(
            time_tr, event_tr, r_tr, t0=t_used,
            n_thresholds=IPCW_YOUDEN_N_THRESHOLDS)

    r_te = risk_at_time_idx(model, x_te, times, t_idx)
    high = r_te >= cut; low = ~high

    def _draw(ax, dur, ev, label, color):
        kmf = KaplanMeierFitter()
        kmf.fit(durations=dur, event_observed=ev)
        sf = kmf.survival_function_
        x = sf.index.values.astype(float)
        y = sf.iloc[:, 0].values.astype(float)
        ax.step(x, y, where="post", color=color, lw=2.0,
                label=label)
        ci = getattr(kmf, "confidence_interval_", None)
        if ci is not None and ci.shape[1] >= 2:
            ax.fill_between(
                x, ci.iloc[:, 0].values, ci.iloc[:, 1].values,
                step="post", color=color, alpha=0.18, linewidth=0)

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    _draw(ax, y_te_model.time.values[low],
          y_te_model.event.values[low],
          f"Low (n={int(low.sum())})", COLOR_DS)
    _draw(ax, y_te_model.time.values[high],
          y_te_model.event.values[high],
          f"High (n={int(high.sum())})", COLOR_RSF)

    cl = "median" if cutoff_type == "median" else "IPCW\u2013Youden"
    ax.set_title(f"{model_label}: KM survival by risk group "
                 f"({cl}, t\u2248{int(round(horizon_req))}y)")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    sns.despine(ax=ax)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=300); plt.close(fig)
    print(f"[INFO] Saved: {out_png}")

def compose_6_panel(
    images: List[str], titles: List[str],
    out_path: str, suptitle: str,
    figsize=(16, 18), dpi=300,
    trim_top_fracs=None, hspace=0.12, top=0.95,
):
    """Compose a 3×2 panel figure from six pre-rendered images."""
    set_journal_style()
    if len(images) != 6 or len(titles) != 6:
        raise ValueError("images and titles must have length 6.")
    if trim_top_fracs is None:
        trim_top_fracs = [0.05] * 6
    miss = [p for p in images if not os.path.exists(p)]
    if miss:
        print(f"[WARN] 6-panel skipped (missing): {miss}")
        return

    imgs = []
    for i, p in enumerate(images):
        img = mpimg.imread(p)
        img = auto_crop_border(img)
        h_ = img.shape[0]
        y0 = int(h_ * trim_top_fracs[i])
        imgs.append(img[y0:, ...])

    fig, axes = plt.subplots(3, 2, figsize=figsize,
                              facecolor="white")
    fig.suptitle(suptitle, fontsize=16, fontweight="bold",
                  fontfamily="Arial", y=0.995)
    letters = ["a)", "b)", "c)", "d)", "e)", "f)"]
    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i]); ax.axis("off")
        ax.set_title(titles[i], fontsize=13, pad=16,
                      fontweight="bold", fontfamily="Arial",
                      loc="center")
        ax.text(0.00, 1.02, letters[i],
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=13, fontweight="bold", fontfamily="Arial")

    plt.subplots_adjust(left=0.02, right=0.99, top=top,
                         bottom=0.015, wspace=0.03, hspace=hspace)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight",
                 facecolor="white")
    plt.close(fig)
    print(f"[INFO] Saved: {out_path}")

# ── Generate single KM plots ──────────────────────────────
CUT_TYPE = "median"
km_paths = {}
for model, label, key in [
    (model_dcph, "DeepSurv", "ds"),
    (model_rsf, "RSF", "rsf"),
]:
    for h in [5.0, 9.0]:
        png = os.path.join(
            args.outdir,
            f"km_single_{key}_{int(h)}yr_{CUT_TYPE}.png")
        plot_km_single(model, label, h, CUT_TYPE, png)
        km_paths[(key, int(h))] = png

# ── 6-panel Figure 1: ROC + Cal + DCA + KM@5y ─────────────
compose_6_panel(
    images=[
        os.path.join(args.outdir,
                     "overlay_roc_deepsurv_vs_rsf.png"),
        os.path.join(args.outdir,
                     "overlay_calibration_deepsurv_vs_rsf.png"),
        os.path.join(args.outdir,
                     "overlay_dca_5yr_deepsurv_vs_rsf.png"),
        os.path.join(args.outdir,
                     "overlay_dca_9yr_deepsurv_vs_rsf.png"),
        km_paths[("ds", 5)],
        km_paths[("rsf", 5)],
    ],
    titles=[
        "ROC at t\u22485y and t\u22489y",
        "Calibration at t\u22485y and t\u22489y",
        "DCA at t\u22485y",
        "DCA at t\u22489y",
        "DeepSurv: KM by risk group (5y)",
        "RSF: KM by risk group (5y)",
    ],
    out_path=os.path.join(args.outdir,
                           "panel6_roc_cal_dca_km5.png"),
    suptitle="DeepSurv vs RSF: discrimination, calibration, "
             "clinical utility, and risk stratification (5y)")

# ── 6-panel Figure 2: AUC-CI + CTD-CI + NB-diff + KM@9y ──
compose_6_panel(
    images=[
        os.path.join(args.outdir,
                     "bootstrap_auc_ci.png"),
        os.path.join(args.outdir,
                     "bootstrap_ctd_ci.png"),
        os.path.join(args.outdir,
                     "nb_diff_5yr.png"),
        os.path.join(args.outdir,
                     "nb_diff_9yr.png"),
        km_paths[("ds", 9)],
        km_paths[("rsf", 9)],
    ],
    titles=[
        "Bootstrap 95% CI for AUC(t)",
        "Bootstrap 95% CI for C-index(t)",
        "NB difference at t\u22485y",
        "NB difference at t\u22489y",
        "DeepSurv: KM by risk group (9y)",
        "RSF: KM by risk group (9y)",
    ],
    out_path=os.path.join(args.outdir,
                           "panel6_auc_ctd_nb_km9.png"),
    suptitle="DeepSurv vs RSF: bootstrap discrimination, "
             "net-benefit difference, and risk stratification (9y)")

# ============================================================
# 17) Export inference artifacts for Streamlit / external use
# ============================================================
if EXPORT_ARTIFACTS:
    try:
        joblib.dump(model_dcph,
                    os.path.join(args.artifacts,
                                 "deepsurv_dcph_model.pkl"))
        joblib.dump(model_rsf,
                    os.path.join(args.artifacts,
                                 "rsf_auton_model.pkl"))
        joblib.dump(transformer,
                    os.path.join(args.artifacts,
                                 "transformer.pkl"))
        joblib.dump(np.asarray(times, dtype=float),
                    os.path.join(args.artifacts, "times.pkl"))

        ranges: Dict[str, Dict[str, float]] = {}
        for col in x_tr_raw.columns:
            s = pd.to_numeric(x_tr_raw[col], errors="coerce")
            if s.notna().sum() == 0:
                continue
            ranges[col] = {
                "min": float(np.nanmin(s.values)),
                "median": float(np.nanmedian(s.values)),
                "max": float(np.nanmax(s.values)),
            }
        with open(os.path.join(args.artifacts,
                                "feature_ranges.json"),
                   "w", encoding="utf-8") as f:
            json.dump(ranges, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Artifacts saved to: "
              f"{os.path.abspath(args.artifacts)}")
    except Exception as e:
        print(f"[WARN] Artifact export failed: {e}")

print(f"\n[DONE] All outputs saved to: "
      f"{os.path.abspath(args.outdir)}")