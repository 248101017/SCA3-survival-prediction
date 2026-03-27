#!/usr/bin/env python3
"""
VBM scatter plots: DeepSurv risk scores vs adjusted grey matter volume
======================================================================
Generates 2x2 panel and individual scatter plots showing the association
between DeepSurv-predicted survival risk scores and regional cerebellar
grey matter volume (GMV) in the external validation cohort.

Usage
-----
    python scatter_vbm.py \
        --clinical data/external_cohort.xlsx \
        --risk     outputs/outer_risk_predictions.csv \
        --outdir   outputs/scatter

If no arguments are provided, default relative paths are used.

Output
------
    - merged_risk_gmv.csv              : merged clinical + risk + adjusted GMV
    - scatter_correlation_summary.csv  : Pearson & Spearman statistics
    - scatter_logscale_*_panel.png     : 2x2 panel figure
    - scatter_logscale_*_{5,9}year.png : individual plots
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import ScalarFormatter

# ── Global plot settings ───────────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "axes.unicode_minus": False,
})

def parse_args():
    """Parse command-line arguments with sensible defaults."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clinical", default="data/external_cohort.xlsx",
                   help="Path to clinical Excel file (external cohort)")
    p.add_argument("--risk", default="outputs/outer_risk_predictions.csv",
                   help="Path to risk prediction CSV")
    p.add_argument("--outdir", default="outputs/scatter",
                   help="Output directory")
    return p.parse_args()

# ── Core functions ─────────────────────────────────────────

def load_and_merge(clinical_path: str, risk_path: str) -> pd.DataFrame:
    """Load clinical and risk files, deduplicate, and merge on id."""
    df_clin = pd.read_excel(clinical_path)
    df_risk = pd.read_csv(risk_path)

    df_clin["id"] = df_clin["id"].astype(str).str.strip()
    df_risk["id"] = df_risk["id"].astype(str).str.strip()

    # Report and remove duplicate IDs
    for label, df in [("clinical", df_clin), ("risk", df_risk)]:
        dups = df[df["id"].duplicated(keep=False)]
        n_dup = len(dups)
        if n_dup > 0:
            dup_ids = dups["id"].unique().tolist()
            print(f"[WARN] {label} file has {n_dup} rows with duplicate IDs: {dup_ids}")

    df_clin = df_clin.drop_duplicates(subset="id", keep="first")
    df_risk = df_risk.drop_duplicates(subset="id", keep="first")

    df = pd.merge(df_clin, df_risk, on="id", how="inner")
    print(f"Merged: {len(df)} subjects "
          f"(clinical: {len(df_clin)}, risk: {len(df_risk)})")

    df = df.rename(columns={
        "Right Cerebellum Crus II":  "gmv_R_CrusII",
        "Left Cerebellum Lobule VI": "gmv_L_LobVI",
    })
    return df

def adjust_gmv(df: pd.DataFrame,
               gmv_cols: list,
               confound_cols: list = None) -> pd.DataFrame:
    """
    Regress out confounders from GMV and return adjusted values
    (residuals + grand mean).
    """
    if confound_cols is None:
        confound_cols = ["age", "sex", "TIV"]

    for c in confound_cols:
        df[c] = df[c].fillna(df[c].median())

    confounds = df[confound_cols].values

    for col in gmv_cols:
        y = df[col].values.astype(float)
        mask = np.isfinite(y)
        reg = LinearRegression().fit(confounds[mask], y[mask])
        df[f"{col}_adj"] = y - reg.predict(confounds) + np.nanmean(y)

    return df

def plot_single(df, x_col, y_col, horizon, region, color, out_path):
    """Generate a single scatter plot with log-scale x-axis."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=300, facecolor="white")

    x = df[x_col].values.astype(float)
    y = df[y_col].values.astype(float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[mask], y[mask]

    ax.scatter(x, y, c=color, s=30, alpha=0.7,
               edgecolors="white", linewidths=0.5, zorder=3)

    # Regression in log space
    log_x = np.log(x)
    slope, intercept, _, _, _ = stats.linregress(log_x, y)
    x_fit_log = np.linspace(log_x.min(), log_x.max(), 200)
    x_fit = np.exp(x_fit_log)
    y_fit = slope * x_fit_log + intercept
    ax.plot(x_fit, y_fit, color="#E74C3C", linewidth=2, zorder=4)

    # 95% confidence band
    n = len(x)
    x_mean = np.mean(log_x)
    se = np.sqrt(np.sum((y - (slope * log_x + intercept)) ** 2) / (n - 2))
    ci = (stats.t.ppf(0.975, n - 2) * se
          * np.sqrt(1 / n + (x_fit_log - x_mean) ** 2
                    / np.sum((log_x - x_mean) ** 2)))
    ax.fill_between(x_fit, y_fit - ci, y_fit + ci,
                     color="#E74C3C", alpha=0.12, zorder=2)

    # Statistics
    r_p, p_p = stats.pearsonr(x, y)
    r_s, p_s = stats.spearmanr(x, y)

    def _fmt_p(p):
        return f"{p:.2e}" if p < 0.001 else f"{p:.3f}"

    stat_text = (f"Pearson r = {r_p:.3f}, p = {_fmt_p(p_p)}\n"
                 f"Spearman \u03c1 = {r_s:.3f}, p = {_fmt_p(p_s)}\n"
                 f"n = {n}")
    ax.text(0.95, 0.95, stat_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=9, fontfamily="Arial",
            bbox=dict(facecolor="white", alpha=0.85,
                      boxstyle="round,pad=0.3", edgecolor="#CCCCCC"))

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.set_xlabel(f"DeepSurv {horizon} risk score (log scale)",
                  fontsize=11, fontfamily="Arial")
    ax.set_ylabel("Adjusted GMV (a.u.)", fontsize=11, fontfamily="Arial")
    ax.set_title(f"{region} ({horizon})",
                 fontsize=12, fontweight="bold", fontfamily="Arial")
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="both", linestyle="--", alpha=0.3, zorder=0)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")

    return {"Region": region, "Horizon": horizon,
            "Pearson_r": r_p, "Pearson_p": p_p,
            "Spearman_rho": r_s, "Spearman_p": p_s, "n": n}

def plot_panel(df, panels, out_path):
    """Generate a 2x2 panel figure."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), dpi=300, facecolor="white")

    for i, (x_col, y_col, horizon, region, letter, color) in enumerate(panels):
        ax = axes.flat[i]

        x = df[x_col].values.astype(float)
        y = df[y_col].values.astype(float)
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
        x, y = x[mask], y[mask]

        ax.scatter(x, y, c=color, s=28, alpha=0.7,
                   edgecolors="white", linewidths=0.5, zorder=3)

        log_x = np.log(x)
        slope, intercept, _, _, _ = stats.linregress(log_x, y)
        x_fit_log = np.linspace(log_x.min(), log_x.max(), 200)
        x_fit = np.exp(x_fit_log)
        y_fit = slope * x_fit_log + intercept
        ax.plot(x_fit, y_fit, color="#E74C3C", linewidth=2, zorder=4)

        n = len(x)
        x_mean = np.mean(log_x)
        se = np.sqrt(np.sum((y - (slope * log_x + intercept)) ** 2) / (n - 2))
        ci = (stats.t.ppf(0.975, n - 2) * se
              * np.sqrt(1 / n + (x_fit_log - x_mean) ** 2
                        / np.sum((log_x - x_mean) ** 2)))
        ax.fill_between(x_fit, y_fit - ci, y_fit + ci,
                         color="#E74C3C", alpha=0.12, zorder=2)

        r_p, p_p = stats.pearsonr(x, y)
        r_s, p_s = stats.spearmanr(x, y)

        def _fmt_p(p):
            return f"{p:.2e}" if p < 0.001 else f"{p:.3f}"

        stat_text = (f"Pearson r = {r_p:.3f}, p = {_fmt_p(p_p)}\n"
                     f"Spearman \u03c1 = {r_s:.3f}, p = {_fmt_p(p_s)}\n"
                     f"n = {n}")
        ax.text(0.95, 0.95, stat_text, transform=ax.transAxes,
                ha="right", va="top", fontsize=8, fontfamily="Arial",
                bbox=dict(facecolor="white", alpha=0.85,
                          boxstyle="round,pad=0.3", edgecolor="#CCCCCC"))

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.get_major_formatter().set_scientific(False)
        ax.set_xlabel(f"DeepSurv {horizon} risk score (log scale)",
                      fontsize=10, fontfamily="Arial")
        ax.set_ylabel("Adjusted GMV (a.u.)", fontsize=10, fontfamily="Arial")
        ax.set_title(f"{letter} {region} ({horizon})",
                     fontsize=11, fontweight="bold", fontfamily="Arial", loc="left")
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="both", linestyle="--", alpha=0.3, zorder=0)

    plt.tight_layout(pad=2.0)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved panel: {out_path}")

# ── Main ───────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load and merge
    df = load_and_merge(args.clinical, args.risk)

    # Adjust GMV for confounders
    df = adjust_gmv(df, gmv_cols=["gmv_R_CrusII", "gmv_L_LobVI"])

    # Save merged data
    merged_csv = os.path.join(args.outdir, "merged_risk_gmv.csv")
    df.to_csv(merged_csv, index=False, encoding="utf-8-sig")
    print(f"Saved merged CSV: {merged_csv}")

    # Panel configuration
    panels = [
        ("deepsurv_risk_5y", "gmv_R_CrusII_adj",
         "5-year", "R Cerebellum Crus II",   "a)", "#2C6FAC"),
        ("deepsurv_risk_9y", "gmv_R_CrusII_adj",
         "9-year", "R Cerebellum Crus II",   "b)", "#E67E22"),
        ("deepsurv_risk_5y", "gmv_L_LobVI_adj",
         "5-year", "L Cerebellum Lobule VI", "c)", "#2C6FAC"),
        ("deepsurv_risk_9y", "gmv_L_LobVI_adj",
         "9-year", "L Cerebellum Lobule VI", "d)", "#E67E22"),
    ]

    # Individual scatter plots
    print("\n--- Individual scatter plots ---")
    summary_rows = []
    for x_col, y_col, horizon, region, letter, color in panels:
        tag = region.replace(" ", "_")
        fname = os.path.join(
            args.outdir,
            f"scatter_logscale_{tag}_{horizon.replace('-', '')}.png"
        )
        row = plot_single(df, x_col, y_col, horizon, region, color, fname)
        summary_rows.append(row)

    # Correlation summary
    df_summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.outdir, "scatter_correlation_summary.csv")
    df_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"Saved summary: {summary_csv}")

    # 2x2 panel figure
    print("\n--- Panel figure ---")
    panel_path = os.path.join(args.outdir, "scatter_logscale_risk_vs_gmv_panel.png")
    plot_panel(df, panels, panel_path)

    # Print summary
    print("\n" + "=" * 70)
    print("CORRELATION SUMMARY")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    print(f"\n[DONE] All outputs saved to: {args.outdir}")

if __name__ == "__main__":
    main()