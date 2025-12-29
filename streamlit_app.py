import os
from pathlib import Path

try:
    import streamlit as st
except Exception as e:
    raise ImportError("streamlit is required to run this app. Install dependencies: `pip install -r requirements.txt`") from e

try:
    import pandas as pd
except Exception as e:
    raise ImportError("pandas is required to run this app. Install dependencies: `pip install -r requirements.txt`") from e

try:
    import joblib
except Exception as e:
    raise ImportError("joblib is required to run this app. Install dependencies: `pip install -r requirements.txt`") from e

# standard library helper for encoding local Lottie JSON if provided
import base64

# Default artifact names (you can replace these)
MODEL_FILE = Path("best_model_with_tuning.joblib")   # change if you saved a different name
ENC_FILE = Path("label_encoder.joblib")


def _find_candidate_artifacts():
    """Search the project for common model/encoder files and return candidate paths."""
    patterns = ["**/*.joblib", "**/*.pkl", "**/*.sav", "**/model*.joblib", "**/model*.pkl"]
    model_paths = []
    enc_paths = []
    for pat in patterns:
        for p in Path('.').rglob(pat):
            # Heuristic: filenames containing 'label' or 'encoder' look like encoders
            name = p.name.lower()
            if "label" in name or "encoder" in name or "le" == p.stem.lower():
                enc_paths.append(p)
            else:
                model_paths.append(p)
    # Deduplicate while preserving order
    def _uniq(seq):
        seen = set(); out = []
        for x in seq:
            if x not in seen:
                seen.add(x); out.append(x)
        return out
    return _uniq(model_paths), _uniq(enc_paths)

@st.cache_resource
def load_artifacts(auto_discover: bool = True):
    """Load model and label encoder safely. If files are missing, attempt to auto-discover candidates

    Returns:
        tuple: (model or None, le or None, error_message or None)
    """
    # If explicit files exist, try them first
    if MODEL_FILE.exists() and ENC_FILE.exists():
        try:
            model = joblib.load(MODEL_FILE)
            le = joblib.load(ENC_FILE)
            return model, le, None
        except Exception as e:
            return None, None, f"Error loading artifacts: {e}"

    if not auto_discover:
        missing = []
        if not MODEL_FILE.exists():
            missing.append(str(MODEL_FILE))
        if not ENC_FILE.exists():
            missing.append(str(ENC_FILE))
        return None, None, f"Missing artifact file(s): {', '.join(missing)}"

    # Try to find candidate files in the repo
    model_candidates, enc_candidates = _find_candidate_artifacts()

    # If no candidates found, return missing message
    if not model_candidates and not enc_candidates:
        missing = [str(MODEL_FILE), str(ENC_FILE)]
        return None, None, f"Missing artifact file(s): {', '.join(missing)}"

    # If running under Streamlit, allow user to pick via sidebar
    chosen_model = None
    chosen_enc = None
    try:
        if isinstance(st, object):
            if model_candidates:
                model_options = [str(p) for p in model_candidates]
                chosen_model = st.sidebar.selectbox("Choose model file", options=model_options, index=0)
            if enc_candidates:
                enc_options = [str(p) for p in enc_candidates]
                chosen_enc = st.sidebar.selectbox("Choose encoder file", options=enc_options, index=0)
    except Exception:
        # Fallback to first candidates if streamlit not active
        chosen_model = str(model_candidates[0]) if model_candidates else None
        chosen_enc = str(enc_candidates[0]) if enc_candidates else None

    # Try loading the chosen files
    try:
        if chosen_model is not None and chosen_enc is not None:
            model = joblib.load(chosen_model)
            le = joblib.load(chosen_enc)
            return model, le, None
        else:
            return None, None, "Model or encoder selection incomplete."
    except Exception as e:
        return None, None, f"Error loading selected artifacts: {e}"

# Attempt to load artifacts (auto-discover if defaults are missing)
model, le, load_err = load_artifacts()


st.set_page_config(page_title="Failure Type Prediction", layout="centered")

# --- Header visuals: optional logo and embedded Lottie animation ---
# Place `logo.png` or `logo.jpg` inside an `assets/` folder or next to this file to show your brand logo in the sidebar.
ASSETS_DIR = Path("assets")
logo_candidates = [ASSETS_DIR / "logo.svg", Path("logo.svg"), ASSETS_DIR / "logo.png", Path("logo.png"), ASSETS_DIR / "logo.jpg", Path("logo.jpg")]
logo_path = next((p for p in logo_candidates if p.exists()), None)

try:
    import streamlit.components.v1 as components
except Exception:
    components = None

if logo_path:
    try:
        # Place the logo in the main header, top-right, sized to ~100px for crispness
        header_left, header_right = st.columns([4,1])
        with header_right:
            if logo_path.suffix.lower() == ".svg":
                svg_bytes = logo_path.read_bytes()
                svg_b64 = base64.b64encode(svg_bytes).decode("ascii")
                html = f'<div style="text-align:right; padding-top:4px;"><img src="data:image/svg+xml;base64,{svg_b64}" style="width:100px; height:auto;" alt="logo"/></div>'
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.image(str(logo_path), width=100, use_column_width=False)
    except Exception:
        # Fallback: show message in top-right area if rendering fails
        header_left, header_right = st.columns([4,1])
        with header_right:
            st.write("Logo failed to render")

st.title("Predictive Maintenance — Failure Type Classification")

st.write("Enter the machine readings, then click **Predict**.")

# Lottie animation (embedded via the Lottie player). If you want a local animation, add `assets/animation.json`.
def _render_lottie(lottie_url: str = None):
    # prefer a local file if present and convert to a data URI
    local_json = ASSETS_DIR / "animation.json"
    lottie_src = lottie_url

    if local_json.exists():
        try:
            raw = local_json.read_bytes()
            b64 = base64.b64encode(raw).decode("ascii")
            lottie_src = f"data:application/json;base64,{b64}"
        except Exception:
            # if encoding fails, fallback to provided URL (or default)
            lottie_src = lottie_url

    # default public animation (maintenance/gear) if no custom URL provided
    if lottie_src is None:
        lottie_src = "https://assets4.lottiefiles.com/packages/lf20_jcikwtux.json"

    if components is None:
        # cannot render without components; show a compact emoji as fallback
        st.markdown("🔧 *Animation unavailable (Streamlit components missing).*")
        return

    html = f"""
    <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
    <lottie-player src="{lottie_src}" background="transparent" speed="1" style="width:100%; height:220px;" loop autoplay></lottie-player>
    """
    components.html(html, height=240)

# Render the animation under the title (non-blocking)
_render_lottie()

# If no logo provided, show a small app name in the sidebar for branding
if not logo_path:
    st.sidebar.markdown("### Predictive Maintenance")
    st.sidebar.write("Upload `assets/logo.png` or `logo.png` to show your logo here.")


# ---- Inputs (edit names ONLY if your CSV columns are different) ----
col1, col2 = st.columns(2)

with col1:
    air = st.number_input("Air temperature [K]", value=300.0)
    proc = st.number_input("Process temperature [K]", value=310.0)
    rot = st.number_input("Rotational speed [rpm]", value=1500.0)

with col2:
    tor = st.number_input("Torque [Nm]", value=40.0)
    wear = st.number_input("Tool wear [min]", value=0.0)
    ptype = st.selectbox("Type", ["L", "M", "H"])

X_new = pd.DataFrame([{
    "Type": ptype,
    "Air temperature [K]": air,
    "Process temperature [K]": proc,
    "Rotational speed [rpm]": rot,
    "Torque [Nm]": tor,
    "Tool wear [min]": wear
}])

st.divider()
st.subheader("Input preview")
st.dataframe(X_new, width='stretch')

# Show artifact load status and provide upload fallback
if load_err:
    st.error(load_err)
    st.info("Place the model and encoder files in the app folder (or update the file names) and reload the app.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Artifacts not found? Upload them here (will be saved locally).**")
    up_model = st.sidebar.file_uploader("Upload model (joblib/pkl)", type=["joblib","pkl","sav","bin","model"], key="up_model")
    up_enc = st.sidebar.file_uploader("Upload encoder (joblib/pkl)", type=["joblib","pkl","sav","bin"], key="up_enc")

    if up_model is not None and up_enc is not None:
        try:
            target_model_path = Path("uploaded_model") / up_model.name if Path("uploaded_model").is_dir() else Path(up_model.name)
            target_enc_path = Path("uploaded_encoder") / up_enc.name if Path("uploaded_encoder").is_dir() else Path(up_enc.name)

            # Save the uploaded files
            with open(target_model_path, "wb") as f:
                f.write(up_model.getbuffer())
            with open(target_enc_path, "wb") as f:
                f.write(up_enc.getbuffer())

            # Try loading
            try:
                _model = joblib.load(target_model_path)
                _le = joblib.load(target_enc_path)
                model = _model
                le = _le
                load_err = None
                st.success("Uploaded artifacts saved and loaded successfully.")
            except Exception as e:
                st.error(f"Uploaded files could not be loaded: {e}")
        except Exception as e:
            st.error(f"Failed to save uploaded artifacts: {e}")
else:
    if st.button("Predict"):
        try:
            pred = model.predict(X_new)
            # Some models return class names (strings) already
            if hasattr(pred, "dtype") and pred.dtype == object:
                pred_name = str(pred[0])
            else:
                # Numeric prediction; prefer to inverse-transform if encoder is available
                if 'le' in locals() and le is not None:
                    pred_int = int(pred[0])
                    pred_name = le.inverse_transform([pred_int])[0]
                else:
                    # fallback: show the raw numeric prediction
                    pred_name = str(pred[0])
            st.success(f"Predicted Failure Type: **{pred_name}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

        # Optional: show probabilities if available
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_new)[0]
                proba_df = pd.DataFrame({"Class": le.classes_, "Probability": proba}).sort_values("Probability", ascending=False)
                st.subheader("Prediction probabilities")
                st.dataframe(proba_df, width='stretch')
            except Exception as e:
                st.warning(f"Could not compute prediction probabilities: {e}")

st.divider()

# ---- Optional: batch prediction from uploaded CSV ----
st.subheader("Batch prediction (upload CSV)")
up = st.file_uploader("Upload a CSV with the same feature columns", type=["csv"])

if up is not None:
    try:
        df_up = pd.read_csv(up)
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        df_up = None

    if df_up is not None:
        needed = set(X_new.columns.tolist())
        missing = needed - set(df_up.columns.tolist())
        if missing:
            st.error(f"Missing columns in uploaded file: {sorted(list(missing))}")
        elif load_err:
            st.error("Model artifacts not loaded — batch prediction is disabled.")
        else:
            try:
                preds = model.predict(df_up[list(X_new.columns)])
                # If preds are not object strings, try to inverse-transform if encoder is present
                if hasattr(preds, "dtype") and preds.dtype == object:
                    preds_names = preds.astype(str)
                else:
                    if 'le' in locals() and le is not None:
                        preds_names = le.inverse_transform(preds.astype(int))
                    else:
                        preds_names = preds.astype(str)

                out_df = df_up.copy()
                out_df["Predicted Failure Type"] = preds_names

                st.subheader("Predictions")
                st.dataframe(out_df.head(20), width='stretch')
                st.download_button(
                    "Download results CSV",
                    data=out_df.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

