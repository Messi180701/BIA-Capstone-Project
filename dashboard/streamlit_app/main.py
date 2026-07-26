import os
from pathlib import Path

import joblib
import numpy as np
import requests
import streamlit as st

st.set_page_config(
    page_title="E-Commerce Customer Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
def get_api_url() -> str:
    env_url = os.getenv("FASTAPI_URL")
    if env_url:
        return env_url.rstrip("/")

    try:
        secret_url = st.secrets.get("FASTAPI_URL")
        if secret_url:
            return str(secret_url).rstrip("/")
    except Exception:
        pass

    return "http://127.0.0.1:8000"


API_URL = get_api_url()

st.markdown(
    """
    <style>
        .stApp {background: #f7f8fc;}
        .block-container {padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1200px;}
        h1, h2, h3 {letter-spacing: -0.02em;}
        .hero {
            padding: 1.7rem 1.9rem;
            border-radius: 22px;
            background: linear-gradient(120deg, #172554 0%, #3730a3 55%, #6d28d9 100%);
            color: white;
            margin-bottom: 1.2rem;
            box-shadow: 0 14px 34px rgba(30, 41, 59, 0.14);
        }
        .hero h1 {color: white; margin: 0; font-size: 2.15rem;}
        .hero p {opacity: 0.88; margin: 0.5rem 0 0; font-size: 1rem;}
        .info-card {
            background: white;
            border: 1px solid #e8eaf2;
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05);
            min-height: 132px;
        }
        .eyebrow {font-size: 0.76rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.08em;}
        .card-title {font-size: 1.05rem; font-weight: 700; margin: 0.25rem 0 0.35rem; color: #111827;}
        .card-copy {font-size: 0.92rem; color: #4b5563; line-height: 1.45;}
        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #e8eaf2;
            padding: 0.9rem 1rem;
            border-radius: 15px;
            box-shadow: 0 5px 16px rgba(15, 23, 42, 0.04);
        }
        div.stButton > button {
            border-radius: 12px;
            height: 3rem;
            font-weight: 700;
            border: 0;
            background: linear-gradient(90deg, #4338ca, #7c3aed);
            color: white;
        }
        div.stButton > button:hover {color: white; border: 0; filter: brightness(1.05);}
        .assistant-answer {
            background: white;
            border-left: 5px solid #6d28d9;
            border-radius: 12px;
            padding: 1rem 1.1rem;
            box-shadow: 0 5px 16px rgba(15, 23, 42, 0.05);
        }
        .small-note {color: #6b7280; font-size: 0.84rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    rf_model = joblib.load(MODELS_DIR / "best_classifier_rf.pkl")
    kmeans_model = joblib.load(MODELS_DIR / "kmeans_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scalar.pkl")
    return rf_model, kmeans_model, scaler


try:
    RF_MODEL, KMEANS_MODEL, SCALER = load_models()
except Exception as error:
    st.error(f"The prediction models could not be loaded: {error}")
    st.stop()


SEGMENT_META = {
    "VIP": {
        "description": "Recent, frequent and high-spending customers.",
        "strategy": "Prioritise premium loyalty benefits, early access and referral rewards.",
    },
    "Loyal": {
        "description": "Regular buyers with consistent engagement and repeat purchases.",
        "strategy": "Use cross-sell campaigns, loyalty points and personalised recommendations.",
    },
    "At Risk": {
        "description": "Customers whose recent activity suggests possible churn.",
        "strategy": "Run targeted win-back offers, reminders and feedback campaigns.",
    },
    "New": {
        "description": "Recently acquired customers with limited purchase history.",
        "strategy": "Use welcome journeys, onboarding offers and second-purchase incentives.",
    },
}


with st.sidebar:
    st.markdown("## 🛍️ Customer Intelligence")
    st.caption("RFM segmentation • ML prediction • AI insights")
    st.divider()
    st.markdown("**Models**")
    st.write("Random Forest classifier")
    st.write("K-Means clustering")
    st.markdown("**Input features**")
    st.write("Recency, Frequency, Monetary")
    st.divider()
    st.markdown("**API connection**")
    st.code(API_URL, language=None)
    st.caption("Set FASTAPI_URL in Streamlit secrets after deployment.")


st.markdown(
    """
    <div class="hero">
        <h1>E-Commerce Customer Intelligence</h1>
        <p>Predict customer value, understand customer segments and receive AI-powered business recommendations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

predictor_tab, assistant_tab, project_tab = st.tabs(
    ["🎯 Customer Predictor", "💬 AI Business Assistant", "📘 Project Overview"]
)

with predictor_tab:
    st.subheader("Predict customer value")
    st.caption("Enter RFM values to estimate customer segment and high-value probability.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            recency = st.number_input(
                "Recency (days)", min_value=0, max_value=1000, value=15,
                help="Number of days since the customer's latest purchase.",
            )
        with col2:
            frequency = st.number_input(
                "Frequency (orders)", min_value=1, max_value=1000, value=12,
                help="Number of unique orders placed by the customer.",
            )
        with col3:
            monetary = st.number_input(
                "Monetary value (£)", min_value=0.0, max_value=500000.0,
                value=8500.0, step=100.0,
                help="Total revenue generated by the customer.",
            )

        submitted = st.form_submit_button("Analyse customer", use_container_width=True)

    if submitted:
        input_data = np.array([[recency, frequency, monetary]])

        prediction = RF_MODEL.predict(input_data)[0]
        probability = float(RF_MODEL.predict_proba(input_data)[0][1])

        transformed_data = np.log1p(input_data)
        scaled_data = SCALER.transform(transformed_data)
        cluster = int(KMEANS_MODEL.predict(scaled_data)[0])

        segment_map = {0: "VIP", 1: "At Risk", 2: "New", 3: "Loyal"}
        segment = segment_map.get(cluster, "Unknown")
        is_high_value = bool(prediction == 1)

        if is_high_value and segment in {"At Risk", "New"}:
            segment = "Loyal"
        elif not is_high_value and segment in {"VIP", "Loyal"}:
            segment = "At Risk"

        st.success("Customer analysis completed")
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Predicted segment", segment)
        metric2.metric("High-value customer", "Yes" if is_high_value else "No")
        metric3.metric("High-value probability", f"{probability * 100:.2f}%")

        st.progress(probability, text="Model confidence toward the high-value class")

        info1, info2 = st.columns(2)
        with info1:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="eyebrow">Customer profile</div>
                    <div class="card-title">{segment}</div>
                    <div class="card-copy">{SEGMENT_META[segment]['description']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with info2:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="eyebrow">Recommended action</div>
                    <div class="card-title">Marketing strategy</div>
                    <div class="card-copy">{SEGMENT_META[segment]['strategy']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("View technical prediction details"):
            st.json(
                {
                    "recency": int(recency),
                    "frequency": int(frequency),
                    "monetary": float(monetary),
                    "kmeans_cluster": cluster,
                    "high_value_probability": round(probability, 4),
                }
            )

with assistant_tab:
    st.subheader("Ask the AI Business Assistant")
    st.caption("The assistant uses Ollama tool calling and verified Python analysis functions.")

    examples = [
        "Which customer segment contributes the most revenue?",
        "How many customers are in each segment?",
        "Where should I spend my marketing budget?",
        "Give me an overview of all customer segments.",
    ]

    selected_example = st.selectbox(
        "Try an example question",
        ["Select an example..."] + examples,
    )
    default_question = "" if selected_example == "Select an example..." else selected_example
    question = st.text_area(
        "Business question",
        value=default_question,
        height=110,
        placeholder="Example: Which segment should receive most of our marketing budget?",
    )

    if st.button("Ask assistant", use_container_width=True):
        if not question.strip():
            st.warning("Enter a business question first.")
        else:
            with st.spinner("Analysing the customer data..."):
                try:
                    response = requests.post(
                        f"{API_URL}/ai-business-assistant",
                        json={"question": question.strip()},
                        timeout=90,
                    )
                    response.raise_for_status()
                    answer = response.json().get("answer", "No answer was returned.")
                    st.markdown(
                        f'<div class="assistant-answer">{answer}</div>',
                        unsafe_allow_html=True,
                    )
                except requests.RequestException as error:
                    st.error(
                        "The AI API could not be reached. Confirm that FastAPI is running "
                        "and FASTAPI_URL points to the deployed backend."
                    )
                    st.caption(str(error))

with project_tab:
    st.subheader("How this solution works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="info-card">
                <div class="eyebrow">1 • Analytics</div>
                <div class="card-title">RFM Segmentation</div>
                <div class="card-copy">Customers are represented through Recency, Frequency and Monetary behaviour.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="info-card">
                <div class="eyebrow">2 • Machine Learning</div>
                <div class="card-title">Prediction & Clustering</div>
                <div class="card-copy">Random Forest predicts high-value status while K-Means assigns customer segments.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="info-card">
                <div class="eyebrow">3 • Generative AI</div>
                <div class="card-title">Business Assistant</div>
                <div class="card-copy">Ollama selects verified analysis tools and explains results in practical business language.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Technology stack")
    st.write("Python • Pandas • Scikit-learn • Streamlit • FastAPI • Ollama Cloud • Power BI")
    st.markdown(
        '<p class="small-note">Built as a capstone project for e-commerce customer segmentation and prediction.</p>',
        unsafe_allow_html=True,
    )
