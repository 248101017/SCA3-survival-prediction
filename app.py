import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

APP_TITLE = "SCA3 Individual Survival Prediction (DeepSurv)"
st.set_page_config(page_title=APP_TITLE, layout="centered")
st.title(APP_TITLE)

# -----------------------------
# Debug (optional)
# -----------------------------
with st.expander("Runtime environment (debug)", expanded=False):
    st.write(f"Python executable: `{sys.executable}`")
    st.write(f"Working directory: `{os.getcwd()}`")

# -----------------------------
# Dependency guard
# -----------------------------
try:
    import auton_survival  # noqa: F401
except Exception as e:
    st.error(
        "Cannot import `auton_survival`, so the model `.pkl` cannot be loaded.\n\n"
        f"Error: `{e}`\n\n"
        "**Recommended CMD:**\n"
        "```bat\n"
        "conda activate sca3train\n"
        "python -m streamlit run app.py --server.port 8501\n"
        "```\n"
    )
    st.stop()

# -----------------------------
# Artifact paths
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

MODEL_PATH = os.path.join(ARTIFACT_DIR, "deepsurv_dcph_model.pkl")
TRANSFORMER_PATH = os.path.join(ARTIFACT_DIR, "transformer.pkl")
TIMES_PATH = os.path.join(ARTIFACT_DIR, "times.pkl")
RANGES_PATH = os.path.join(ARTIFACT_DIR, "feature_ranges.json")

# Optional cut-off files (training median; if present, take priority)
CUTOFF_5Y_PATH = os.path.join(ARTIFACT_DIR, "cutoffs_train_median_req5y.json")
CUTOFF_9Y_PATH = os.path.join(ARTIFACT_DIR, "cutoffs_train_median_req9y.json")

for p, label in [
    (MODEL_PATH, "model"),
    (TRANSFORMER_PATH, "transformer"),
    (TIMES_PATH, "times"),
]:
    if not os.path.exists(p):
        st.error(f"Missing {label} artifact: `{p}`")
        st.stop()

# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_resource
def load_artifacts(model_path, transformer_path, times_path, ranges_path):
    model = joblib.load(model_path)
    transformer = joblib.load(transformer_path)

    times = np.asarray(joblib.load(times_path), dtype=float)
    times = np.sort(np.unique(times))

    ranges = None
    if os.path.exists(ranges_path):
        with open(ranges_path, "r", encoding="utf-8") as f:
            ranges = json.load(f)

    return model, transformer, times, ranges

def _load_cutoffs(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

model, transformer, times, ranges = load_artifacts(
    MODEL_PATH, TRANSFORMER_PATH, TIMES_PATH, RANGES_PATH
)

# Fallback to your provided constants (DeepSurv median cutoffs)
cut5 = _load_cutoffs(CUTOFF_5Y_PATH) or {
    "horizon_req_years": 5.0,
    "grid_used_years": 4.75878048,
    "cutoff_type": "median",
    "DeepSurv_cutoff": 0.07171018072329877,
}
cut9 = _load_cutoffs(CUTOFF_9Y_PATH) or {
    "horizon_req_years": 9.0,
    "grid_used_years": 9.3813304,
    "cutoff_type": "median",
    "DeepSurv_cutoff": 0.27662956232446095,
}

with st.expander("Loaded artifacts (for reproducibility)", expanded=False):
    st.write(f"- model: `{MODEL_PATH}`")
    st.write(f"- transformer: `{TRANSFORMER_PATH}`")
    st.write(f"- times: `{TIMES_PATH}`")
    st.write(f"- feature ranges: `{RANGES_PATH if os.path.exists(RANGES_PATH) else 'not found'}`")
    st.write(f"- times grid: n={len(times)}, min={times.min():.3f}, max={times.max():.3f}")
    st.write("**Training-derived median cut-points used for risk grouping (DeepSurv):**")
    st.write(
        f"- t≈5y (grid {float(cut5['grid_used_years']):.3f}y): cutoff={float(cut5['DeepSurv_cutoff']):.4f}\n"
        f"- t≈9y (grid {float(cut9['grid_used_years']):.3f}y): cutoff={float(cut9['DeepSurv_cutoff']):.4f}"
    )

# -----------------------------
# Feature contract
# -----------------------------
RAW_NUM = [
    "BMI",
    "disease_duration",
    "ATXN3_CAG_Long",
    "SARA_Total",
    "EQ_VAS",
    "PHQ_Depression",
    "GAD7_Anxiety",
]
RAW_BIN = [
    "INAS_Muscle_atrophy",
    "INAS_Fasciculations",
    "INAS_Sensory_symptoms",
]
PROC_CANONICAL = [
    "BMI",
    "disease_duration",
    "ATXN3_CAG_Long",
    "SARA_Total",
    "EQ_VAS",
    "PHQ_Depression",
    "GAD7_Anxiety",
    "INAS_Muscle_atrophy_1",
    "INAS_Fasciculations_1",
    "INAS_Sensory_symptoms_1",
]

DISPLAY_NAME_RAW = {
    "BMI": "BMI (kg/m²)",
    "disease_duration": "Disease duration (years)",
    "ATXN3_CAG_Long": "Long CAG repeats",
    "SARA_Total": "SARA score",
    "EQ_VAS": "EQ-VAS",
    "PHQ_Depression": "PHQ-9 depression score",
    "GAD7_Anxiety": "GAD-7 anxiety score",
    "INAS_Muscle_atrophy": "INAS muscle atrophy",
    "INAS_Fasciculations": "INAS fasciculations",
    "INAS_Sensory_symptoms": "INAS sensory symptoms",
}

DISPLAY_NAME_PROC = {
    "BMI": "BMI (kg/m²)",
    "disease_duration": "Disease duration (years)",
    "ATXN3_CAG_Long": "Long CAG repeats",
    "SARA_Total": "SARA score",
    "EQ_VAS": "EQ-VAS",
    "PHQ_Depression": "PHQ-9 depression score",
    "GAD7_Anxiety": "GAD-7 anxiety score",
    "INAS_Muscle_atrophy_1": "INAS muscle atrophy",
    "INAS_Fasciculations_1": "INAS fasciculations",
    "INAS_Sensory_symptoms_1": "INAS sensory symptoms",
}

def _disp(col: str) -> str:
    return DISPLAY_NAME_RAW.get(col, col)

def _hint(name: str) -> str:
    if not ranges or name not in ranges:
        return ""
    r = ranges[name]
    return f"(train min/median/max: {r['min']:.2f}/{r['median']:.2f}/{r['max']:.2f})"

# -----------------------------
# Inputs UI
# -----------------------------
st.subheader("Patient inputs")
with st.form("input_form"):
    st.markdown("### Numeric features")
    inputs = {}

    for f in RAW_NUM:
        default = float(ranges[f]["median"]) if ranges and f in ranges else 0.0
        inputs[f] = st.number_input(
            f"{_disp(f)} {_hint(f)}",
            value=float(default),
            step=0.1,
            format="%.2f",
        )

    st.markdown("### Binary clinical features (0 = absent / 1 = present)")
    for f in RAW_BIN:
        default_bin = int(round(float(ranges[f]["median"]))) if ranges and f in ranges else 0
        default_bin = 1 if default_bin >= 1 else 0
        inputs[f] = st.selectbox(f"{_disp(f)} {_hint(f)}", [0, 1], index=default_bin)

    st.markdown("### Prediction horizons")
    show_5y = st.checkbox("Show 5-year survival / risk", value=True)
    show_9y = st.checkbox("Show 9-year survival / risk", value=True)

    submitted = st.form_submit_button("▶  Predict survival curve")

df_input = pd.DataFrame([inputs], columns=RAW_NUM + RAW_BIN)
df_input[RAW_NUM] = df_input[RAW_NUM].astype(float)
df_input[RAW_BIN] = df_input[RAW_BIN].astype(int)

st.write("**Input preview:**")
st.dataframe(df_input.rename(columns=DISPLAY_NAME_RAW), hide_index=True)

# OOD warning
if ranges:
    ood_msgs = []
    for f in RAW_NUM:
        v = float(df_input.loc[0, f])
        mn = float(ranges[f]["min"])
        mx = float(ranges[f]["max"])
        if v < mn or v > mx:
            ood_msgs.append(f"{_disp(f)} = {v:.2f} (training range: [{mn:.2f}, {mx:.2f}])")
    if ood_msgs:
        st.warning(
            "Out-of-distribution inputs detected — predictions may be less reliable:\n- "
            + "\n- ".join(ood_msgs)
        )

# -----------------------------
# Helpers
# -----------------------------
def _closest_idx(arr: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(arr - float(t))))

def _get_model_input_dim(m) -> int | None:
    try:
        tm = m._model.torch_model
        if hasattr(tm, "embedding") and hasattr(tm.embedding, "__getitem__"):
            layer0 = tm.embedding[0]
            if hasattr(layer0, "in_features"):
                return int(layer0.in_features)
        if hasattr(tm, "expert") and hasattr(tm.expert, "in_features"):
            return int(tm.expert.in_features)
    except Exception:
        pass
    return None

def _ensure_processed_feature_contract(X_proc, raw_df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(X_proc, np.ndarray):
        X_proc = pd.DataFrame(X_proc)
    elif not isinstance(X_proc, pd.DataFrame):
        try:
            X_proc = pd.DataFrame(X_proc)
        except Exception:
            raise ValueError(f"Unsupported transformer output type: {type(X_proc)}")

    # Try to set column names if transformer output has integer columns
    if X_proc.columns is None or any(isinstance(c, int) for c in X_proc.columns):
        if X_proc.shape[1] == 7:
            X_proc.columns = RAW_NUM
        elif X_proc.shape[1] == 10:
            X_proc.columns = PROC_CANONICAL

    cols = list(X_proc.columns)

    if all(c in cols for c in PROC_CANONICAL):
        return X_proc[PROC_CANONICAL].copy()

    # Only numeric returned
    if all(c in cols for c in RAW_NUM) and X_proc.shape[1] == 7:
        out = X_proc[RAW_NUM].copy()
        out["INAS_Muscle_atrophy_1"] = raw_df["INAS_Muscle_atrophy"].astype(int).values
        out["INAS_Fasciculations_1"] = raw_df["INAS_Fasciculations"].astype(int).values
        out["INAS_Sensory_symptoms_1"] = raw_df["INAS_Sensory_symptoms"].astype(int).values
        return out[PROC_CANONICAL].copy()

    # Raw numeric + raw binary returned
    if all(c in cols for c in RAW_NUM + RAW_BIN):
        out = X_proc[RAW_NUM + RAW_BIN].copy()
        out["INAS_Muscle_atrophy_1"] = out["INAS_Muscle_atrophy"].astype(int)
        out["INAS_Fasciculations_1"] = out["INAS_Fasciculations"].astype(int)
        out["INAS_Sensory_symptoms_1"] = out["INAS_Sensory_symptoms"].astype(int)
        out = out.drop(columns=RAW_BIN)
        return out[PROC_CANONICAL].copy()

    # Fallback: create missing columns from raw
    out = X_proc.copy()
    for c in PROC_CANONICAL:
        if c not in out.columns:
            if c.endswith("_1"):
                raw_name = c.replace("_1", "")
                out[c] = raw_df[raw_name].astype(int).values if raw_name in raw_df.columns else 0
            else:
                out[c] = raw_df[c].values if c in raw_df.columns else 0

    return out[PROC_CANONICAL].copy()

def _risk_group(risk: float, cutoff: float) -> str:
    return "High risk" if float(risk) >= float(cutoff) else "Low risk"

# -----------------------------
# Predict
# -----------------------------
if submitted:
    try:
        X_proc_raw = transformer.transform(df_input)
        X_proc = _ensure_processed_feature_contract(X_proc_raw, df_input)

        expected_dim = _get_model_input_dim(model)
        got_dim = int(X_proc.shape[1])

        if expected_dim is not None and expected_dim != got_dim:
            st.error(
                f"Feature dimension mismatch: model expects {expected_dim} but prepared {got_dim}.\n\n"
                "Re-export artifacts from the same training run."
            )
            st.stop()

        surv = model.predict_survival(X_proc, times)
        surv_curve = np.asarray(surv[0], dtype=float)

        # ---- Plot survival curve ----
        st.subheader("Predicted survival curve")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.step(times, surv_curve, where="post", color="#1f77b4", linewidth=2)
        ax.set_xlabel("Time (years)", fontsize=12)
        ax.set_ylabel("Survival probability", fontsize=12)
        ax.set_title("Individualised predicted survival curve", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        st.pyplot(fig)

        # ---- Key horizons with risk groups ----
        st.subheader("Survival probability, risk, and risk group at key horizons")

        requested = []
        if show_5y:
            requested.append(5.0)
        if show_9y:
            requested.append(9.0)

        rows = []
        for t0 in requested:
            if t0 > float(times.max()):
                rows.append(
                    {
                        "Horizon (years)": t0,
                        "Grid used (years)": "Out of range",
                        "Survival probability S(t)": "—",
                        "Risk 1−S(t)": "—",
                        "Risk group (median cut-off)": "Out of range",
                    }
                )
                continue

            idx = _closest_idx(times, t0)
            t_used = float(times[idx])
            s_t = float(surv_curve[idx])
            risk_t = float(1.0 - s_t)

            if int(round(t0)) == 5:
                cutoff = float(cut5["DeepSurv_cutoff"])
                grid_label = float(cut5.get("grid_used_years", t_used))
            elif int(round(t0)) == 9:
                cutoff = float(cut9["DeepSurv_cutoff"])
                grid_label = float(cut9.get("grid_used_years", t_used))
            else:
                cutoff = float("nan")
                grid_label = t_used

            group = _risk_group(risk_t, cutoff) if np.isfinite(cutoff) else "—"

            rows.append(
                {
                    "Horizon (years)": int(round(t0)),
                    "Grid used (years)": f"{grid_label:.3f}",
                    "Survival probability S(t)": f"{s_t:.2%}",
                    "Risk 1−S(t)": f"{risk_t:.2%}",
                    "Risk group (median cut-off)": f"{group} (cut-off={cutoff:.4f})",
                }
            )

        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)

        # ---- Optional: common time point summary ----
        st.subheader("Summary table (common time points)")
        common = [1, 3, 5, 7, 9]
        rows2 = []
        for t in common:
            if t <= float(times.max()):
                i = _closest_idx(times, t)
                rows2.append(
                    {
                        "Time point (years)": t,
                        "Grid point used (years)": round(float(times[i]), 3),
                        "Survival probability S(t)": f"{float(surv_curve[i]):.2%}",
                        "Risk 1−S(t)": f"{float(1.0 - surv_curve[i]):.2%}",
                    }
                )
        if rows2:
            st.dataframe(pd.DataFrame(rows2), hide_index=True)

        st.success("Prediction complete.")

    except Exception as e:
        import traceback
        st.error(f"Prediction failed: {str(e)}")
        with st.expander("Full traceback"):
            st.code(traceback.format_exc())

with st.expander("Troubleshooting (important)", expanded=False):
    st.markdown(
        """
**If you see feature dimension mismatch**  
Re-export artifacts from the same training run (model + transformer + times).

**If deployment fails**  
Ensure `runtime.txt` pins Python 3.10 and `requirements.txt` contains `torch` and `auton-survival`.
"""
    )