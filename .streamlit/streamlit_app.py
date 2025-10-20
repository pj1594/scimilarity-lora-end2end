#Streamlit cellsimilarity app

import streamlit as st, requests, json, pandas as pd

st.set_page_config(page_title="SCimilarity LoRA — Single-Cell Classifier", layout="wide")

# ------------- CONFIG -------------
# Put your Cloud Run FastAPI /predict URL here (or set via Space secret; see Step 3)
# ---- Robust API URL resolver for HF Spaces + local ----
import os
try:
    import streamlit as st
except Exception:
    st = None

def resolve_api_url(default="https://nasir-spacious-kamila.ngrok-free.dev/predict"):
    # 1) HF Spaces also exposes repo secrets as environment variables
    env = os.getenv("API_URL")
    if env:
        return env

    # 2) Try Streamlit secrets if available
    try:
        if st is not None and hasattr(st, "secrets") and "API_URL" in st.secrets:
            return st.secrets["API_URL"]
    except Exception:
        pass

    # 3) Fallback (works locally too)
    return default

API_URL = resolve_api_url()
# Optional: show where it came from for debugging
if st is not None:
    st.sidebar.info(f"Backend: `{API_URL}`")
# -------------------------------------------------------

TEMP_DEFAULT = 0.7
THRESH_DEFAULT = 0.4
# ----------------------------------

st.title("🧬 SCimilarity + LoRA — Single-Cell Classifier (UI)")
st.caption("This Streamlit UI calls a FastAPI backend (Cloud Run) for predictions.")

left, right = st.columns([2,1], gap="large")

with left:
    st.subheader("Input expression (JSON)")
    default_json = {
        "CD19": 8.5, "MS4A1": 9.2, "CD79A": 7.8,
        "CD3D": 0.1, "CD8A": 0.0, "CD4": 0.0,
        "MKI67": 0.1, "GAPDH": 6.2
    }
    txt = st.text_area("Edit gene:value JSON", value=json.dumps(default_json, indent=2), height=260)

with right:
    st.subheader("Settings")
    TEMP = st.slider("Temperature (server-side optional)", 0.3, 2.0, TEMP_DEFAULT, 0.1)
    THRESH = st.slider("Unknown threshold (client label policy)", 0.0, 0.9, THRESH_DEFAULT, 0.05)
    st.info("These sliders are UI hints; the backend may have its own default TEMP/THRESH.")

st.markdown("---")
btn = st.button("🔎 Predict", use_container_width=True)

def _validate_json(js_txt: str):
    try:
        payload = json.loads(js_txt)
        assert isinstance(payload, dict), "Top-level JSON must be a dict of gene: value."
        # ensure values are numeric
        for k, v in payload.items():
            _ = float(v)
        return payload, None
    except Exception as e:
        return None, f"Invalid JSON: {e}"

if btn:
    expr, err = _validate_json(txt)
    if err:
        st.error(err)
    else:
        with st.spinner("Calling backend…"):
            try:
                resp = requests.post(
                    API_URL,
                    json={"expression": expr},
                    timeout=60
                )
                resp.raise_for_status()
                out = resp.json()

                # Render result
                st.success(f"Prediction: **{out.get('cell_type','?')}**  |  Confidence: **{out.get('confidence',0.0):.2f}**")
                st.caption("Raw response:")
                st.json(out)

                # Optional Top-k support if backend includes 'probs' & 'classes'
                if "probs" in out and "classes" in out:
                    import numpy as np
                    probs = out["probs"]
                    classes = out["classes"]
                    order = list(reversed(sorted(range(len(probs)), key=lambda i: probs[i])))[:5]
                    df = pd.DataFrame({
                        "rank": list(range(1, len(order)+1)),
                        "cell_type": [classes[i] for i in order],
                        "probability": [float(probs[i]) for i in order],
                    })
                    st.subheader("Top-5 candidates")
                    st.table(df)
                else:
                    st.info("Top-k table not shown: backend did not return 'probs'/'classes'. (Optional feature.)")

            except Exception as e:
                st.error(f"Backend error: {e}")
                st.write("Check that API_URL is correct and reachable.")
