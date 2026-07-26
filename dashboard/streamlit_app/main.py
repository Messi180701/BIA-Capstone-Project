
import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Customer Value Predictor", page_icon="🛍️", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))
# main.py lives in dashboard/fastapi_app/ ; the trained models live in <repo_root>/models/
MODELS_DIR = os.path.normpath(os.path.join(BASE, "..", "..", "models"))

@st.cache_resource
def load_models():
    rf = joblib.load(os.path.join(MODELS_DIR, "best_classifier_rf.pkl"))
    kmeans = joblib.load(os.path.join(MODELS_DIR, "kmeans_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scalar.pkl"))
    return rf, kmeans, scaler

try:
    RF_MODEL, KMEANS_MODEL, SCALER = load_models()
except Exception as e:
    st.error(f"Failed to load models:\n{e}")
    st.stop()

SEGMENT_META = {
    "VIP": {
        "description": "Highest value customers. Recent, frequent, high spenders.",
        "strategy": "Exclusive loyalty rewards, early access, personal account manager.",
    },
    "Loyal": {
        "description": "Regular buyers with consistent purchases.",
        "strategy": "Upsell campaigns and loyalty benefits.",
    },
    "At Risk": {
        "description": "Customers showing signs of churn.",
        "strategy": "Win-back offers and discounts.",
    },
    "New": {
        "description": "Newly acquired customers.",
        "strategy": "Welcome offers and onboarding campaigns.",
    },
}

st.title("🛍️ Customer Value Predictor")
st.caption("Random Forest + KMeans based Customer Value Prediction")

col1, col2 = st.columns(2)

with col1:
    recency = st.number_input("Recency (days)", 0, 1000, 15)
    frequency = st.number_input("Frequency", 1, 1000, 12)

with col2:
    monetary = st.number_input("Monetary (£)", 0.0, 500000.0, 8500.0)

if st.button("Predict", use_container_width=True):
    X = np.array([[recency, frequency, monetary]])

    pred = RF_MODEL.predict(X)[0]
    prob = RF_MODEL.predict_proba(X)[0][1]

    X_log = np.log1p(X)
    X_scaled = SCALER.transform(X_log)

    cluster = int(KMEANS_MODEL.predict(X_scaled)[0])

    segment_map = {
        0: "VIP",
        1: "At Risk",
        2: "New",
        3: "Loyal"
    }

    segment = segment_map.get(cluster, "Unknown")
    high = bool(pred == 1)

    if high and segment in ["At Risk", "New"]:
        segment = "Loyal"

    if not high and segment in ["VIP", "Loyal"]:
        segment = "At Risk"

    st.success("Prediction Complete")

    a,b,c = st.columns(3)

    a.metric("Segment", segment)
    b.metric("High Value", "Yes" if high else "No")
    c.metric("Probability", f"{prob*100:.2f}%")

    st.progress(float(prob))

    st.subheader("Customer Insight")
    st.write(f"**Description:** {SEGMENT_META[segment]['description']}")
    st.write(f"**Recommended Strategy:** {SEGMENT_META[segment]['strategy']}")

    with st.expander("Technical Details"):
        st.json({
            "Recency": recency,
            "Frequency": frequency,
            "Monetary": monetary,
            "Cluster": cluster,
            "High Value Probability": round(float(prob),4)
        })

st.sidebar.header("Model Information")
st.sidebar.write("Algorithm: Random Forest")
st.sidebar.write("Segmentation: KMeans")
st.sidebar.write("Features: Recency, Frequency, Monetary")
