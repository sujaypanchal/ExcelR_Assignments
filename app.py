import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load the trained model and scaler
@st.cache_resource
def load_model():
    with open('diabetes_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# App configuration
st.set_page_config(page_title="Diabetes Prediction", layout="wide")
st.title("🩺 Diabetes Prediction App")

# Sidebar for inputs
st.sidebar.header("Patient Information")
pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
glucose = st.sidebar.slider('Glucose', 0, 200, 120)
blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 70)
skin_thickness = st.sidebar.slider('Skin Thickness', 0, 99, 20)
insulin = st.sidebar.slider('Insulin', 0, 846, 79)
bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
dpf = st.sidebar.slider('Diabetes Pedigree', 0.078, 2.42, 0.47)
age = st.sidebar.slider('Age', 21, 81, 33)

# Create input dataframe
input_data = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [dpf],
    'Age': [age]
})

# Display inputs
st.subheader("📋 Patient Input")
st.write(input_data)

# Make prediction
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)
probability = model.predict_proba(input_scaled)[0][1]

# Show results
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔮 Prediction Result")
    if prediction[0] == 1:
        st.error(f"⚠️ **High Risk of Diabetes**")
        st.write(f"Probability: {probability*100:.1f}%")
        st.progress(probability)
    else:
        st.success(f"✅ **Low Risk of Diabetes**")
        st.write(f"Probability: {probability*100:.1f}%")
        st.progress(probability)

with col2:
    st.subheader("📊 Risk Factors")
    if glucose > 140:
        st.warning("High Glucose Level")
    if bmi > 30:
        st.warning("High BMI (Obese)")
    if age > 45:
        st.warning("Age over 45")
    if glucose <= 140 and bmi <= 30 and age <= 45:
        st.info("No significant risk factors")

# Model info
with st.expander("ℹ️ About This Model"):
    st.write("""
    This model uses Logistic Regression trained on the Pima Indians Diabetes Dataset.
    
    **Model Performance:**
    - Accuracy: ~77%
    - AUC-ROC: ~83%
    """)