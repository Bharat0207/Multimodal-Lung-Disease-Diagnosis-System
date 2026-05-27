import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import joblib
import os
import shap
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================

st.set_page_config(page_title="Disease Prediction", layout="wide")

st.title("🏥 Multimodal Disease Prediction System")
st.markdown("Real predictions using trained models (GRU + MLP + DenseNet + Fusion)")

TARGET_LABELS = [
    "Pneumonia",
    "Pleural Effusion",
    "Edema",
    "Consolidation"
]

PROJECT_ROOT = Path(__file__).resolve().parent
DEMO_DIR = PROJECT_ROOT / "demo_dataset"

# =========================
# FEATURE NAMES (CLEAN)
# =========================

FEATURE_NAMES = [
    "HR Mean", "RR Mean", "Temp Mean", "SBP Mean",
    "DBP Mean", "WBC Mean", "Neutrophils Mean", "CRP Mean",

    "HR Std", "RR Std", "Temp Std", "SBP Std",
    "DBP Std", "WBC Std", "Neutrophils Std", "CRP Std",

    "HR Min", "RR Min", "Temp Min", "SBP Min",
    "DBP Min", "WBC Min", "Neutrophils Min", "CRP Min",

    "HR Max", "RR Max", "Temp Max", "SBP Max",
    "DBP Max", "WBC Max", "Neutrophils Max", "CRP Max"
]

SHAP_FEATURE_NAMES = np.array([
    "GRU Pneumonia", "GRU Effusion", "GRU Edema", "GRU Consolidation",
    "MLP Pneumonia", "MLP Effusion", "MLP Edema", "MLP Consolidation",
    "X-ray Pneumonia", "X-ray Effusion", "X-ray Edema", "X-ray Consolidation"
])

# =========================
# LOAD DATA
# =========================

@st.cache_resource
def load_demo_data():
    df = pd.read_csv(DEMO_DIR / "demo.csv")
    timeseries = np.load(DEMO_DIR / "demo_timeseries.npy")
    return df, timeseries

@st.cache_resource
def load_model_outputs():
    gru = np.load(PROJECT_ROOT / "gru_preds.npy")
    mlp = np.load(PROJECT_ROOT / "mlp_preds.npy")
    dense = np.load(PROJECT_ROOT / "dense_patient_preds.npy")
    return gru, mlp, dense

@st.cache_resource
def load_fusion_models():
    models = {}
    for label in TARGET_LABELS:
        models[label] = joblib.load(PROJECT_ROOT / f"fusion_{label}.pkl")
    thresholds = np.load(PROJECT_ROOT / "fusion_thresholds.npy")
    return models, thresholds

demo_df, demo_timeseries = load_demo_data()
gru_preds, mlp_preds, dense_preds = load_model_outputs()
fusion_models, thresholds = load_fusion_models()

st.success(f"✅ Demo dataset loaded: {demo_df.shape[0]} patients")

# =========================
# SIDEBAR
# =========================

st.sidebar.header("👤 Patient Selection")

selected_idx = st.sidebar.selectbox(
    "Select Patient",
    range(len(demo_df)),
    format_func=lambda x: f"Patient {x}"
)

patient_row = demo_df.iloc[selected_idx]
patient_ts = demo_timeseries[selected_idx]

# =========================
# UI
# =========================

col1, col2 = st.columns(2)

# IMAGE
with col1:
    st.subheader("📸 Chest X-Ray")
    img_path = patient_row["image_path"]

    if os.path.exists(img_path):
        img = Image.open(img_path)
        st.image(img, width="stretch")
    else:
        st.warning("Image not found")

# FEATURES
with col2:
    st.subheader("📊 Patient Features")

    feature_cols = [c for c in patient_row.index if c.startswith("feat_")]

    if len(feature_cols) == 32:
        for i in range(8):  # show means only for clean UI
            st.write(f"{FEATURE_NAMES[i]}: {patient_row[feature_cols[i]]:.3f}")

    st.write("### ⏱ Timeseries Info")
    st.write("Shape:", patient_ts.shape)
    st.write("Mean:", round(patient_ts.mean(), 3))
    st.write("Std:", round(patient_ts.std(), 3))

# =========================
# PREDICTION
# =========================

st.markdown("---")
st.subheader("🔮 Disease Prediction")

if st.button("Run Prediction"):

    original_idx = int(patient_row["original_idx"])

    g = gru_preds[original_idx]
    m = mlp_preds[original_idx]
    d = dense_preds[original_idx] * 2.5

    X = np.concatenate([g, m, d]).reshape(1, -1)

    predictions = {}

    for i, label in enumerate(TARGET_LABELS):

        model = fusion_models[label]

        prob = model.predict_proba(X)[0][1]
        threshold = float(thresholds[i])
        binary = int(prob >= threshold)

        gt = int(patient_row[label])

        predictions[label] = {
            "prob": prob,
            "threshold": threshold,
            "binary": binary,
            "gt": gt
        }

    st.session_state["run"] = True
    st.session_state["X"] = X
    st.session_state["predictions"] = predictions

# =========================
# SHOW RESULTS
# =========================

if "run" in st.session_state:

    predictions = st.session_state["predictions"]

    st.success("Prediction complete!")

    cols = st.columns(2)

    for i, label in enumerate(TARGET_LABELS):

        with cols[i % 2]:

            data = predictions[label]

            emoji = "🔴" if data["binary"] else "🟢"

            st.markdown(f"### {emoji} {label}")

            c1, c2 = st.columns(2)
            c1.metric("Probability", f"{data['prob']*100:.1f}%")
            c2.metric("Threshold", f"{data['threshold']:.2f}")

            st.progress(float(data["prob"]))

            match = "✅" if data["binary"] == data["gt"] else "❌"
            st.caption(f"Ground Truth: {data['gt']} {match}")

    # =========================
    # SHAP (POSITIVE ONLY + SLIM)
    # =========================

    st.markdown("---")
    st.subheader("🔍 Risk Drivers (SHAP)")

    try:
        X = st.session_state["X"]

        selected_label = st.selectbox(
            "Select Disease to Explain",
            TARGET_LABELS
        )

        model = fusion_models[selected_label]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)[0]

        # Positive only
        pos_mask = shap_values > 0

        if np.sum(pos_mask) == 0:
            st.warning("No strong positive risk drivers.")
        else:
            pos_values = shap_values[pos_mask]
            pos_features = SHAP_FEATURE_NAMES[pos_mask]

            # Top 5
            top_k = min(5, len(pos_values))
            top_idx = np.argsort(pos_values)[-top_k:]

            features = pos_features[top_idx]
            values = pos_values[top_idx]

            # Plot
            fig, ax = plt.subplots(figsize=(8, 5))

            ax.barh(features, values, height=0.35, color="green")

            ax.set_title(f"{selected_label} Risk Drivers", fontsize=12)
            ax.set_xlabel("Contribution to Risk")

            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)

            # Text explanation
            st.write("### 🧠 Key Risk Factors")

            for i in reversed(range(len(features))):
                st.write(f"• **{features[i]}** increases risk")

    except Exception as e:
        st.error(f"SHAP error: {str(e)}")

# =========================
# FOOTER
# =========================

st.markdown("---")
st.caption("⚠️ AI-assisted predictions (for research/demo use only)")