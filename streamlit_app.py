from pathlib import Path
import base64

import streamlit as st
import pandas as pd

# Optional ML dependency (only needed if you want prediction)
try:
    import joblib
except Exception:
    joblib = None

# -------------------- App Config --------------------
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

# -------------------- Optional Assets --------------------
ASSETS_DIR = Path("assets")
logo_candidates = [
    ASSETS_DIR / "logo.svg", Path("logo.svg"),
    ASSETS_DIR / "logo.png", Path("logo.png"),
    ASSETS_DIR / "logo.jpg", Path("logo.jpg"),
]
logo_path = next((p for p in logo_candidates if p.exists()), None)

try:
    import streamlit.components.v1 as components
except Exception:
    components = None


def render_lottie(lottie_url: str | None = None) -> None:
    local_json = ASSETS_DIR / "animation.json"
    lottie_src = lottie_url

    if local_json.exists():
        try:
            raw = local_json.read_bytes()
            b64 = base64.b64encode(raw).decode("ascii")
            lottie_src = f"data:application/json;base64,{b64}"
        except Exception:
            pass

    if lottie_src is None:
        lottie_src = "https://assets4.lottiefiles.com/packages/lf20_jcikwtux.json"

    if components is None:
        st.markdown("🔧 *Animation unavailable.*")
        return

    html = f"""
    <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
    <lottie-player src="{lottie_src}" background="transparent" speed="1"
      style="width:100%; height:200px;" loop autoplay></lottie-player>
    """
    components.html(html, height=220)


# -------------------- Optional Model Artifacts --------------------
MODEL_FILE = Path("best_model_with_tuning.joblib")
ENC_FILE = Path("label_encoder.joblib")


@st.cache_resource
def load_model_artifacts(model_path: str, enc_path: str):
    if joblib is None:
        raise RuntimeError("joblib is not available. Install dependencies.")
    model = joblib.load(model_path)
    le = joblib.load(enc_path)
    return model, le


def try_load_artifacts():
    if not MODEL_FILE.exists() or not ENC_FILE.exists():
        return None, None, "Model artifacts not found (prediction will be disabled)."
    try:
        model, le = load_model_artifacts(str(MODEL_FILE), str(ENC_FILE))
        return model, le, None
    except Exception as e:
        return None, None, f"Failed to load model artifacts: {e}"


def safe_predict_proba(model, X: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba
    return None


def ensure_columns(df: pd.DataFrame, needed: list[str]) -> tuple[bool, list[str]]:
    missing = [c for c in needed if c not in df.columns]
    return (len(missing) == 0), missing


def compute_urgency(severity_weight: float, confidence: float) -> float:
    score = severity_weight * confidence
    return max(0.0, min(1.0, float(score)))


def urgency_level(score: float) -> str:
    if score >= 0.75:
        return "CRITICAL"
    if score >= 0.50:
        return "HIGH"
    if score >= 0.25:
        return "MEDIUM"
    return "LOW"


def infer_failure_column(df: pd.DataFrame) -> str | None:
    candidates = ["Failure Type", "failure_type", "Target", "target", "label", "Label"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


# -------------------- UI Header --------------------
left, right = st.columns([7, 1])
with left:
    st.title("Predictive Maintenance — Dataset Dashboard")
    st.caption("Upload dataset.csv to show key tables and insights. Predictions are enabled only if model artifacts exist.")
with right:
    if logo_path:
        try:
            if logo_path.suffix.lower() == ".svg":
                svg_b64 = base64.b64encode(logo_path.read_bytes()).decode("ascii")
                st.markdown(
                    f'<div style="text-align:right;"><img src="data:image/svg+xml;base64,{svg_b64}" style="width:90px;"/></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.image(str(logo_path), width=90)
        except Exception:
            pass

render_lottie()

st.divider()

# -------------------- Upload CSV --------------------
st.subheader("1) Upload dataset.csv")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is None:
    st.info("Please upload a dataset CSV to begin.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

# -------------------- Sidebar Controls --------------------
st.sidebar.header("Dashboard Controls")

preview_rows = st.sidebar.slider("Preview rows", min_value=5, max_value=100, value=20, step=5)
show_describe = st.sidebar.checkbox("Show numeric summary (describe)", value=True)
show_missing = st.sidebar.checkbox("Show missing values table", value=True)
show_duplicates = st.sidebar.checkbox("Show duplicates info", value=True)
show_target_dist = st.sidebar.checkbox("Show target distribution (if available)", value=True)
show_correlation = st.sidebar.checkbox("Show correlation table (numeric)", value=False)

st.sidebar.markdown("---")
enable_prediction = st.sidebar.checkbox("Enable model prediction (if artifacts exist)", value=True)
unknown_threshold = st.sidebar.slider("Unknown threshold (min confidence)", 0.10, 0.95, 0.55, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("Severity Weights (editable)")
default_severity = {
    "No Failure": 0.0,
    "Heat Dissipation Failure": 0.75,
    "Power Failure": 0.85,
    "Overstrain Failure": 0.90,
    "Tool Wear Failure": 0.65,
    "Random Failures": 0.80,
}
severity = {}
for k, v in default_severity.items():
    severity[k] = st.sidebar.slider(k, 0.0, 1.0, float(v), 0.05)

# -------------------- Quick Overview --------------------
st.subheader("2) Quick Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", f"{df.shape[1]:,}")
c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
c4.metric("Memory (approx.)", f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

st.write("### Dataset preview")
st.dataframe(df.head(preview_rows), width="stretch")

# -------------------- Missing Values --------------------
if show_missing:
    st.subheader("3) Missing Values")
    miss = df.isna().sum().sort_values(ascending=False)
    miss_df = pd.DataFrame({"Column": miss.index, "MissingCount": miss.values})
    miss_df["MissingPercent"] = (miss_df["MissingCount"] / len(df) * 100).round(2)
    st.dataframe(miss_df, width="stretch")

# -------------------- Duplicates --------------------
if show_duplicates:
    st.subheader("4) Duplicates")
    dup_count = int(df.duplicated().sum())
    st.write(f"Duplicate rows: **{dup_count:,}**")
    if dup_count > 0:
        st.dataframe(df[df.duplicated()].head(preview_rows), width="stretch")

# -------------------- Summary Stats --------------------
if show_describe:
    st.subheader("5) Numeric Summary (describe)")
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] == 0:
        st.info("No numeric columns detected.")
    else:
        st.dataframe(numeric_df.describe().T, width="stretch")

# -------------------- Target / Failure Distribution --------------------
target_col = infer_failure_column(df)

if show_target_dist:
    st.subheader("6) Target Distribution")
    if target_col is None:
        st.info("No target column found. Expected a column like 'Failure Type' or 'target'.")
    else:
        vc = df[target_col].value_counts(dropna=False)
        dist = pd.DataFrame({"Class": vc.index.astype(str), "Count": vc.values})
        dist["Percent"] = (dist["Count"] / len(df) * 100).round(2)
        st.dataframe(dist, width="stretch")

# -------------------- Correlation (optional) --------------------
if show_correlation:
    st.subheader("7) Correlation (numeric)")
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        st.info("Need at least 2 numeric columns to compute correlation.")
    else:
        corr = numeric_df.corr(numeric_only=True)
        st.dataframe(corr, width="stretch")
