#!/usr/bin/env python3
"""
Data preprocessing for SCA3 survival modelling
===============================================
Reads an SPSS .sav file, applies academic variable renaming,
performs missing-rate filtering, near-zero variance screening,
KNN/mode imputation, mixed-type correlation matrix computation,
and exports all intermediate results.

Usage
-----
    python data_preprocessing.py
    python data_preprocessing.py --input data/raw.sav --outdir outputs/lasso

Outputs
-------
    - missing_report.csv
    - filtered_variables_summary.csv
    - features_imputed_academic.csv
    - corr_matrix_academic.csv / .png / .pdf
    - strong_corr_pairs.csv / .txt
    - run_summary.txt

Dependencies
------------
    numpy, pandas, matplotlib, seaborn, scipy, scikit-learn, pyreadstat
"""

import os
import argparse
import warnings
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import seaborn as sns

from scipy.stats import pointbiserialr, chi2_contingency
from sklearn.impute import KNNImputer

warnings.filterwarnings("ignore")

# ── Global plot defaults ───────────────────────────────────
plt.rcParams.update({
    "font.family": ["Times New Roman", "serif"],
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "axes.titleweight": "normal",
    "axes.unicode_minus": False,
})
sns.set_theme(style="white")

# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/data.sav",
                   help="Path to SPSS .sav file")
    p.add_argument("--outdir", default="outputs/lasso",
                   help="Output directory")
    return p.parse_args()

# Outcome columns
TIME_COL_RAW  = "Survival_time"
EVENT_COL_RAW = "Primary_outcome"
TIME_COL_OUT  = "Survival_time"
EVENT_COL_OUT = "status"

# Pipeline parameters
MISSING_THRESH  = 0.10
NZV_MAJORITY    = 0.90
DISCRETE_MAX_UQ = 5
KNN_K           = 5
CORR_CUTOFF     = 0.70

# Heatmap parameters
HM_FIGSIZE    = (18, 18)
HM_CMAP       = "viridis"
HM_XTICK_ROT  = 75
HM_TICK_FONT  = 10
HM_NUM_FONT   = 10
HM_TOP_MARGIN = 0.78
HM_XTICK_PAD  = 8

# Academic-to-raw variable mapping
MAPPING = {
    "Sex": "Sex",
    "Age at baseline": "Age_at_visit",
    "Long CAG repeats": "ATXN3_CAG_Long",
    "Disease duration": "disease_duration",
    "Age of onset": "Age_of_onset",
    "BMI": "BMI",
    "INAS count": "INAS_Total_score",
    "SARA score": "SARA_Total",
    "Functional stage": "Functional_stage",
    "UHDRS score": "UHDRS_Total_score",
    "EQ-VAS": "EQ_VAS",
    "Barthel index": "Barthel_Index",
    "PHQ-9 depression": "PHQ_Depression",
    "GAD-7 anxiety": "GAD7_Anxiety",
    "MoCA Score": "MoCA",
    "SCAFI score": "SCAFI",
    "INAS_Hyperreflexia": "INAS_Hyperreflexia",
    "INAS_Arreflexia": "INAS_Arreflexia",
    "INAS_Extensor plantar response": "INAS_Extensor_plantar",
    "INAS_Spasticity": "INAS_Spasticity",
    "INAS_Paresis": "INAS_Paresis",
    "INAS_Muscle atrophy": "INAS_Muscle_atrophy",
    "INAS_Fasciculations": "INAS_Fasciculations",
    "INAS_Myoclonus": "INAS_Myoclonus",
    "INAS_Rigidity": "INAS_Rigidity",
    "INAS_Chorea dyskinesia": "INAS_Chorea_dyskinesia",
    "INAS_Dystonia": "INAS_Dystonia",
    "INAS_Resting tremor": "INAS_Resting_tremor",
    "INAS_Sensory symptoms": "INAS_Sensory_symptoms",
    "INAS_Urinary dysfunction": "INAS_Urinary_dysfunction",
    "INAS_Cognitive impairment": "INAS_Cognitive_impairment",
    "INAS_Oculomotor signs": "INAS_Brain_oculomotor_signs",
}

ACAD_VARS = list(MAPPING.keys())
RAW_VARS  = list(MAPPING.values())

# Grouped variable order for heatmap
GROUP_ORDER = {
    "Demographics": ["Sex", "Age at baseline", "BMI"],
    "Genetics / Disease history": [
        "Long CAG repeats", "Age of onset", "Disease duration"],
    "Clinical scales": [
        "SARA score", "Functional stage", "UHDRS score",
        "Barthel index", "EQ-VAS", "MoCA Score", "SCAFI score"],
    "Mood": ["PHQ-9 depression", "GAD-7 anxiety"],
    "INAS subitems": [
        "INAS_Hyperreflexia", "INAS_Arreflexia",
        "INAS_Extensor plantar response", "INAS_Spasticity",
        "INAS_Paresis", "INAS_Muscle atrophy",
        "INAS_Fasciculations", "INAS_Myoclonus",
        "INAS_Rigidity", "INAS_Chorea dyskinesia",
        "INAS_Dystonia", "INAS_Resting tremor",
        "INAS_Sensory symptoms", "INAS_Urinary dysfunction",
        "INAS_Cognitive impairment", "INAS_Oculomotor signs"],
    "INAS total": ["INAS count"],
}

# ═══════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════

def to_event01(x: pd.Series) -> pd.Series:
    """Convert event column to binary 0/1."""
    if pd.api.types.is_numeric_dtype(x):
        if set(x.dropna().unique()).issubset({0, 1}):
            return x.astype(int)
    xs = x.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=x.index, dtype=float)
    out[xs.isin(["1", "yes", "y", "true", "event", "case"])] = 1
    out[xs.isin(["0", "no", "n", "false", "censor", "censored",
                 "control", "non-event"])] = 0
    num = pd.to_numeric(xs, errors="coerce")
    out[out.isna() & num.notna()] = num[out.isna() & num.notna()]
    if not set(out.dropna().unique()).issubset({0, 1}):
        raise ValueError(f"Cannot convert event column to 0/1: "
                         f"unique = {out.dropna().unique()}")
    return out.astype(int)

def mode_impute(s: pd.Series) -> pd.Series:
    """Fill missing values with the mode."""
    if s.isna().sum() == 0:
        return s
    modes = s.dropna().mode()
    return s.fillna(modes.iloc[0]) if len(modes) > 0 else s

def is_discrete(s: pd.Series) -> bool:
    """Check if a series is discrete (object/category or few unique)."""
    if s.dtype == "object" or pd.api.types.is_categorical_dtype(s):
        return True
    return s.nunique(dropna=True) <= DISCRETE_MAX_UQ

def majority_proportion(s: pd.Series) -> float:
    """Proportion of the most frequent value among non-missing."""
    s = s.dropna()
    if len(s) == 0:
        return np.nan
    vc = s.value_counts()
    return float(vc.iloc[0] / vc.sum())

def pearson_corr(x, y):
    return float(pd.to_numeric(x, errors="coerce")
                 .corr(pd.to_numeric(y, errors="coerce")))

def pb_corr(cont, cat):
    """Point-biserial correlation between continuous and binary."""
    cont = pd.to_numeric(cont, errors="coerce")
    codes = cat.astype("category").cat.codes.replace(-1, np.nan)
    if cat.astype("category").nunique(dropna=True) != 2:
        return np.nan
    mask = cont.notna() & codes.notna()
    if mask.sum() < 3:
        return np.nan
    return float(pointbiserialr(cont[mask], codes[mask])[0])

def cramers_v(x, y):
    """Cramer's V for two categorical variables."""
    ct = pd.crosstab(x.astype("category"), y.astype("category"))
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan
    chi2 = chi2_contingency(ct)[0]
    n = ct.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))

def load_sav(path: str, outdir: str) -> tuple:
    """Load SPSS .sav file; tries pyreadstat first, then R fallback."""
    try:
        import pyreadstat
        df, _ = pyreadstat.read_sav(path)
        return df, "pyreadstat"
    except Exception:
        pass

    temp_csv = os.path.join(outdir, "_tmp_sav.csv")
    r_code = (
        'suppressMessages(library(haven))\n'
        'suppressMessages(library(readr))\n'
        f'df <- haven::read_sav("{path.replace(chr(92), "/")}")\n'
        f'readr::write_csv(as.data.frame(df), "{temp_csv.replace(chr(92), "/")}")\n'
    )
    r_script = os.path.join(outdir, "_tmp_read_sav.R")
    with open(r_script, "w") as f:
        f.write(r_code)
    proc = subprocess.run(["Rscript", r_script],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Cannot load .sav file. Install pyreadstat or ensure "
            "R + haven are available.\n" + proc.stderr)
    return pd.read_csv(temp_csv), "R(haven)"

# ═══════════════════════════════════════════════════════════
# Heatmap
# ═══════════════════════════════════════════════════════════

def plot_heatmap(corr_matrix, ordered_cols, save_path,
                 fig_size=HM_FIGSIZE, tick_fs=HM_TICK_FONT,
                 num_fs=HM_NUM_FONT, cmap_name=HM_CMAP,
                 xtick_rot=HM_XTICK_ROT, bold_cutoff=CORR_CUTOFF,
                 top_margin=HM_TOP_MARGIN, xtick_pad=HM_XTICK_PAD):
    """
    Mixed-type correlation heatmap.
    Upper triangle: circles (radius proportional to |corr|).
    Lower triangle: numeric labels (bold if |corr| >= cutoff).
    """
    M_df = corr_matrix.loc[ordered_cols, ordered_cols]
    labels = M_df.columns.tolist()
    M = M_df.values.astype(float)
    n = len(labels)
    cmap = plt.get_cmap(cmap_name)

    fig, ax = plt.subplots(figsize=fig_size)
    plt.subplots_adjust(top=top_margin, right=0.90)

    for i in range(n):
        for j in range(n):
            val = M[i, j]
            if np.isnan(val):
                continue
            x, y = j, n - 1 - i

            if i == j:
                ax.add_patch(plt.Circle((x, y), 0.40,
                             color=cmap(0.99), alpha=1.0))
            elif i < j:
                radius = 0.45 * abs(val)
                ax.add_patch(plt.Circle((x, y), radius,
                             color=cmap((val + 1) / 2), alpha=0.85))
            else:
                fw = "bold" if abs(val) >= bold_cutoff else "normal"
                ax.text(x, y, f"{val:.2f}", ha="center", va="center",
                        fontsize=num_fs, fontweight=fw,
                        color=cmap((val + 1) / 2),
                        path_effects=[path_effects.withStroke(
                            linewidth=1.5, foreground="white")])

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.xaxis.tick_top()
    ax.set_xticklabels(labels, fontsize=tick_fs, rotation=xtick_rot, ha="left")
    ax.set_yticklabels(labels[::-1], fontsize=tick_fs)
    ax.tick_params(axis="x", length=0, pad=xtick_pad)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="lightgray", linestyle="-",
            linewidth=0.5, alpha=0.5)
    ax.grid(which="major", visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cax = fig.add_axes([0.92, 0.15, 0.02, 0.65])
    sm = cm.ScalarMappable(cmap=cmap,
                           norm=mcolors.Normalize(-1, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(size=0, labelsize=tick_fs)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Heatmap saved: {save_path}")

# ═══════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════

def main():
    args = parse_args()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # ── Load data ──────────────────────────────────────────
    df_raw, load_method = load_sav(args.input, outdir)
    print(f"[INFO] Loaded via {load_method}: {df_raw.shape}")

    required = [TIME_COL_RAW, EVENT_COL_RAW] + RAW_VARS
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise ValueError("Missing columns:\n" +
                         "\n".join(f"  - {c}" for c in missing))

    df = df_raw[[TIME_COL_RAW, EVENT_COL_RAW] + RAW_VARS].copy()
    df[TIME_COL_RAW] = pd.to_numeric(df[TIME_COL_RAW], errors="coerce")
    df[EVENT_COL_RAW] = to_event01(df[EVENT_COL_RAW])

    # ── Rename to academic names ───────────────────────────
    rename_map = {v: k for k, v in MAPPING.items()}
    df_acad = df.rename(columns=rename_map)
    df_acad = df_acad.rename(columns={EVENT_COL_RAW: EVENT_COL_OUT})

    raw_csv = os.path.join(outdir, "features_raw_academic.csv")
    df_acad[[TIME_COL_OUT, EVENT_COL_OUT] + ACAD_VARS].to_csv(
        raw_csv, index=False, encoding="utf-8-sig")

    # ── Missing rate filter ────────────────────────────────
    X = df_acad[ACAD_VARS].copy()
    miss_rate = X.isna().mean().sort_values(ascending=False)

    miss_report = pd.DataFrame({
        "feature": miss_rate.index,
        "missing_rate": miss_rate.values,
        "missing_pct": (miss_rate.values * 100).round(2),
    })
    miss_report.to_csv(os.path.join(outdir, "missing_report.csv"),
                       index=False, encoding="utf-8-sig")

    keep = miss_rate[miss_rate <= MISSING_THRESH].index.tolist()
    dropped_miss = miss_rate[miss_rate > MISSING_THRESH].index.tolist()
    X = X[keep].copy()
    print(f"[INFO] Dropped (missingness > {MISSING_THRESH:.0%}): "
          f"{dropped_miss}")

    # ── Near-zero variance filter ──────────────────────────
    dropped_nzv = []
    nzv_rows = []
    for col in X.columns:
        s = X[col]
        disc = is_discrete(s)
        maj = majority_proportion(s) if disc else np.nan
        flag = disc and not np.isnan(maj) and maj >= NZV_MAJORITY
        nzv_rows.append({
            "feature": col,
            "is_discrete": disc,
            "n_unique": int(s.nunique(dropna=True)),
            "majority_prop": maj,
            "dropped": flag,
        })
        if flag:
            dropped_nzv.append(col)

    pd.DataFrame(nzv_rows).to_csv(
        os.path.join(outdir, "nzv_screening.csv"),
        index=False, encoding="utf-8-sig")

    X = X.drop(columns=dropped_nzv, errors="ignore")
    print(f"[INFO] Dropped (NZV, majority >= {NZV_MAJORITY}): "
          f"{dropped_nzv}")

    # ── Summary of filtered variables ──────────────────────
    filt_summary = (
        [{"feature": f, "reason": "missingness"} for f in dropped_miss] +
        [{"feature": f, "reason": "near_zero_variance"} for f in dropped_nzv]
    )
    pd.DataFrame(filt_summary).to_csv(
        os.path.join(outdir, "filtered_variables_summary.csv"),
        index=False, encoding="utf-8-sig")

    # ── Imputation ─────────────────────────────────────────
    num_cols = [c for c in X.columns
                if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    for c in cat_cols:
        X[c] = mode_impute(X[c].astype("category"))
    if num_cols:
        imp = KNNImputer(n_neighbors=KNN_K)
        X[num_cols] = pd.DataFrame(
            imp.fit_transform(X[num_cols].apply(
                pd.to_numeric, errors="coerce")),
            columns=num_cols, index=X.index)

    if X.isna().any().any():
        raise ValueError("NAs remain after imputation.")

    df_out = pd.concat(
        [df_acad[[TIME_COL_OUT, EVENT_COL_OUT]], X], axis=1)
    imputed_csv = os.path.join(outdir, "features_imputed_academic.csv")
    df_out.to_csv(imputed_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Imputed features: {X.shape[1]} variables, "
          f"{X.shape[0]} subjects")

    # ── Mixed-type correlation matrix ──────────────────────
    cols = list(X.columns)
    n = len(cols)
    corr = pd.DataFrame(np.nan, index=cols, columns=cols)

    for i in range(n):
        for j in range(i, n):
            xa, xb = X[cols[i]], X[cols[j]]
            na = pd.api.types.is_numeric_dtype(xa)
            nb = pd.api.types.is_numeric_dtype(xb)

            if na and nb:
                v = pearson_corr(xa, xb)
            elif na and not nb:
                v = pb_corr(xa, xb)
            elif not na and nb:
                v = pb_corr(xb, xa)
            else:
                v = cramers_v(xa, xb)

            corr.iloc[i, j] = v
            corr.iloc[j, i] = v

    corr.to_csv(os.path.join(outdir, "corr_matrix_academic.csv"),
                encoding="utf-8-sig")

    # ── Heatmap ────────────────────────────────────────────
    ordered = []
    for _, group_cols in GROUP_ORDER.items():
        ordered.extend([c for c in group_cols if c in corr.columns])
    ordered.extend([c for c in corr.columns if c not in ordered])

    for ext in ["pdf", "png"]:
        plot_heatmap(
            corr, ordered,
            os.path.join(outdir, f"corr_heatmap_academic.{ext}"))

    # ── Strong correlation pairs ───────────────────────────
    pairs = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            v = corr.iloc[i, j]
            if pd.notna(v) and abs(v) >= CORR_CUTOFF:
                pairs.append({
                    "var1": corr.index[i],
                    "var2": corr.columns[j],
                    "corr": float(v),
                    "abs_corr": float(abs(v)),
                })

    pairs_df = pd.DataFrame(pairs)
    if len(pairs_df):
        pairs_df = pairs_df.sort_values("abs_corr", ascending=False)

    pairs_df.to_csv(
        os.path.join(outdir, f"strong_corr_pairs_ge{CORR_CUTOFF}.csv"),
        index=False, encoding="utf-8-sig")

    with open(os.path.join(outdir, f"strong_corr_pairs_ge{CORR_CUTOFF}.txt"),
              "w") as f:
        f.write(f"Strong pairs (|corr| >= {CORR_CUTOFF})\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        for _, r in pairs_df.iterrows():
            f.write(f"{r['var1']}  <->  {r['var2']}  "
                    f"corr = {r['corr']:.4f}\n")

    # ── Run summary ────────────────────────────────────────
    with open(os.path.join(outdir, "run_summary.txt"), "w") as f:
        f.write("Data preprocessing summary\n")
        f.write(f"{'='*40}\n")
        f.write(f"Input:              {args.input}\n")
        f.write(f"Output:             {outdir}\n")
        f.write(f"Load method:        {load_method}\n")
        f.write(f"Missing threshold:  {MISSING_THRESH}\n")
        f.write(f"NZV threshold:      {NZV_MAJORITY}\n")
        f.write(f"KNN k:              {KNN_K}\n")
        f.write(f"Corr cutoff:        {CORR_CUTOFF}\n")
        f.write(f"Dropped (missing):  {dropped_miss}\n")
        f.write(f"Dropped (NZV):      {dropped_nzv}\n")
        f.write(f"Retained features:  {X.shape[1]}\n")
        f.write(f"Strong pairs:       {len(pairs_df)}\n")

    print(f"\n[DONE] All outputs saved to: {outdir}")

if __name__ == "__main__":
    main()