# diabetes_app.py
# 🌿 Diabetes Prediction App (Random Forest) — with password protection

import streamlit as st
import pandas as pd
import joblib
import os

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

# ---------- PASSWORD PROTECTION ----------
st.title("🔒 Secure Access")

PASSWORD = "Millicent2025"  # ✅ change this to your desired password

# Simple password input
password = st.text_input("Enter Password to Access App:", type="password")

if password != PASSWORD:
    st.warning("Please enter the correct password to access the prediction app.")
    st.stop()

# ---------- LOAD TRAINED MODEL ----------
@st.cache_resource
def load_model():
    model_path = "RandomForest_model.pkl"  # Ensure model file is in the same folder or adjust path
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: {model_path}. Please check your repo structure.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# ---------- HEADER ----------
st.title("🌿 Diabetes Prediction App (Random Forest)")
st.write("This app uses your trained Random Forest model to predict diabetes based on patient details.")

# ---------- INPUT FORM ----------
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("AGE", min_value=1, max_value=120, value=30)
    gender = st.selectbox("GENDER", ['F', 'M'])
    weight = st.number_input("WEIGHT (kg)", min_value=0.0, max_value=300.0, value=70.0)

with col2:
    bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=25.0)
    blood_sugar = st.number_input("BLOOD SUGAR (mmol/L)", min_value=0.0, max_value=40.0, value=8.0)
    htn = st.selectbox("HTN (Hypertension)", [0.0, 1.0])

with col3:
    systolic_bp = st.number_input("SYSTOLIC BP (mmHg)", min_value=0, max_value=300, value=120)
    diastolic_bp = st.number_input("DIASTOLIC BP (mmHg)", min_value=0, max_value=200, value=80)
    treatment = st.selectbox(
        "TREATMENT",
        [
            'a', 'ab', 'abe', 'ae', 'ade', 'e', 'ad',
            'aec', 'ace', 'ce', 'ebe', 'aw', 'ac'
        ],
        index=0
    )

# ---------- PREPARE INPUT DATA ----------
input_dict = {
    'AGE': [age],
    'GENDER': [gender],
    'WEIGHT(kg)': [weight],
    'BMI': [bmi],
    'BLOOD SUGAR(mmol/L)': [blood_sugar],
    'HTN': [htn],
    'SYSTOLIC BP': [systolic_bp],
    'DIASTOLIC BP': [diastolic_bp],
    'TREATMENT': [treatment]
}

input_df = pd.DataFrame(input_dict)

# One-hot encode categorical variables
input_df = pd.get_dummies(input_df, columns=['GENDER', 'TREATMENT'], drop_first=True)

# Align columns with model’s expected input
expected_cols = [
    'AGE', 'WEIGHT(kg)', 'BMI', 'BLOOD SUGAR(mmol/L)',
    'HTN', 'SYSTOLIC BP', 'DIASTOLIC BP', 'GENDER_M'
]

for col in expected_cols:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[expected_cols]

# ---------- PREDICTION ----------
if st.button("Predict"):
    try:
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.subheader("🔍 Prediction Result")
        outcome_mapping = {0: "Non-Diabetic", 1: "Diabetic"}
        st.write("**Outcome:**", outcome_mapping.get(pred, "Unknown"))
        st.write("**Probability of Diabetes:**", f"{prob:.3f}")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

# ---------- FOOTER ----------
st.markdown("---")
st.caption("Developed by Millicent Chesang | Powered by AI & Data Analytics for Public Health")
