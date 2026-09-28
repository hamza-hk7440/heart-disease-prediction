import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Risk Prediction Tool")
st.write("Enter patient clinical parameters to calculate the estimated probability of heart disease.")

# Load Model & Scaler
@st.cache_resource
def load_assets():
    model = joblib.load('best_logistic_regression_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
    
    # Input Form
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=55)
        sex = st.selectbox("Sex", ["Female", "Male"])
        trestbps = st.number_input("Resting BP (mm Hg)", min_value=50, max_value=250, value=130)
        chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=240)
        thalch = st.number_input("Max Heart Rate (thalach)", min_value=50, max_value=220, value=140)

    with col2:
        cp = st.selectbox("Chest Pain Type", ["Non-Anginal Pain", "Atypical Angina", "Typical Angina"])
        fbs = st.checkbox("Fasting Blood Sugar > 120 mg/dL")
        restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality"])
        exang = st.checkbox("Exercise Induced Angina")
        slope = st.selectbox("ST Slope", ["Upsloping", "Flat"])
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    if st.button("Calculate Risk Score", type="primary"):
        # Build patient dataframe with exact one-hot encoded features
        patient_data = pd.DataFrame([{
            'age': age,
            'trestbps': float(trestbps),
            'chol': float(chol),
            'thalch': float(thalch),
            'oldpeak': float(oldpeak),
            'sex_Male': 1 if sex == "Male" else 0,
            'cp_atypical angina': 1 if cp == "Atypical Angina" else 0,
            'cp_non-anginal': 1 if cp == "Non-Anginal Pain" else 0,
            'cp_typical angina': 1 if cp == "Typical Angina" else 0,
            'fbs_True': 1 if fbs else 0,
            'restecg_normal': 1 if restecg == "Normal" else 0,
            'restecg_st-t abnormality': 1 if restecg == "ST-T Wave Abnormality" else 0,
            'exang_True': 1 if exang else 0,
            'slope_flat': 1 if slope == "Flat" else 0,
            'slope_upsloping': 1 if slope == "Upsloping" else 0
            
        }])

        # Scale features and run prediction
        scaled_input = scaler.transform(patient_data)
        prediction = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0][1] * 100

        st.divider()
        if prediction == 1:
            st.error(f"⚠️ **High Risk Detected** — Estimated Probability: **{prob:.2f}%**")
        else:
            st.success(f"✅ **Low Risk / Healthy** — Estimated Probability: **{prob:.2f}%**")

except Exception as e:
    st.error(f"Error loading model files: {e}")