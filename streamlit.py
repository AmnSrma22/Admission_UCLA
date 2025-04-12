import streamlit as st
import pandas as pd
import pickle
from src.data.make_dataset import load_and_preprocess_data
from src.features.build_features import build_features
from src.models.train_model import train_model
from src.models.predict_model import evaluate_model
from src.models.visualization import plot_residuals
from sklearn.preprocessing import MinMaxScaler

# ----- Streamlit Page Config -----
st.set_page_config(page_title="UCLA Admission Predictor", layout="wide")

# ----- Inject Custom CSS -----
st.markdown("""
    <style>
        .big-font {
            font-size: 28px !important;
        }
        .small-font {
            font-size: 16px !important;
            color: #666;
        }
        .result-box {
            padding: 1.5rem;
            border-radius: 10px;
            background-color: #f0f2f6;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
    </style>
""", unsafe_allow_html=True)

# ----- Title Section -----
st.markdown("<h1 class='big-font'>🎓 UCLA Admission Chance Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='small-font'>Enter your academic profile to estimate your admission probability.</p>", unsafe_allow_html=True)
st.markdown("---")

# ----- Sidebar Model Training -----
with st.sidebar:
    st.header("🧠 Train the Model")
    model, scaler = None, None

    if st.button("🚀 Train Model"):
        with st.spinner("Training in progress..."):
            df = load_and_preprocess_data("dataset.csv")
            X = build_features(df)
            y = df['Admit_Chance']
            model, scaler, X_test_scaled, y_test = train_model(X, y)
            rmse, r2 = evaluate_model(model, X_test_scaled, y_test)
            st.success("✅ Training complete!")
            st.metric("RMSE", f"{rmse:.4f}")
            st.metric("R²", f"{r2:.4f}")
            st.markdown("### Residual Plot")
            plot_residuals(y_test, model.predict(X_test_scaled))

# ----- Load Model if Exists -----
try:
    with open("models/nn_model.pkl", "rb") as f:
        model = pickle.load(f)
    df = load_and_preprocess_data("dataset.csv")
    X = build_features(df)
    scaler = MinMaxScaler().fit(X)
except:
    st.warning("⚠️ Please train the model first using the sidebar.")

# ----- User Input Form -----
with st.container():
    st.subheader("📋 Applicant Information")

    with st.form("admission_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            GRE = st.slider("📈 GRE Score", 260, 340, 320)
            TOEFL = st.slider("🗣️ TOEFL Score", 0, 120, 110)
        with col2:
            UR = st.selectbox("🏛️ University Rating", range(1, 6), index=2)
            SOP = st.slider("📝 SOP Strength", 1.0, 5.0, 3.5)
        with col3:
            LOR = st.slider("📜 LOR Strength", 1.0, 5.0, 3.5)
            CGPA = st.slider("🎓 CGPA", 0.0, 10.0, 8.5)

        Research = st.radio("🔬 Research Experience", [0, 1], horizontal=True, format_func=lambda x: "Yes" if x else "No")

        submitted = st.form_submit_button("🎯 Predict Admission")

# ----- Prediction Output -----
if submitted:
    if model and scaler:
        input_df = pd.DataFrame([{
            'GRE_Score': GRE,
            'TOEFL_Score': TOEFL,
            'University_Rating': UR,
            'SOP': SOP,
            'LOR': LOR,
            'CGPA': CGPA,
            'Research': Research
        }])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        percent = prediction * 100

        st.markdown("---")
        st.subheader("📊 Prediction Result")
        col_result, col_meter = st.columns([2, 1])

        with col_result:
            st.markdown(f"<div class='result-box'><h2>🎓 Admission Chance: {percent:.2f}%</h2></div>", unsafe_allow_html=True)

        with col_meter:
            st.progress(min(int(percent), 100), text="Likelihood")

    else:
        st.error("❌ Model not available. Please train the model first.")
