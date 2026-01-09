import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.title("Heart Failure Prediction")
st.write("Let's check the heart condition of the patient")
st.write("Please fill the requested details below")

# ================= LOAD MODELS =================
with open("sex_encode.pkl", "rb") as f:
    sex_encode = pickle.load(f)

with open("chestpain.pkl", "rb") as f:
    ChestPainType_encode = pickle.load(f)

with open("RestingECG_encode.pkl", "rb") as f:
    RestingECG_encode = pickle.load(f)

with open("ExerciseAngina_encode.pkl", "rb") as f:
    ExerciseAngina_encode = pickle.load(f)

with open("ST_Slope_encode.pkl", "rb") as f:
    ST_Slope_encode = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ================= USER INPUT =================
age = st.slider("Age", 1, 100, 45)
sex = st.selectbox("Gender", sex_encode.classes_)
chestpain = st.selectbox("Chest Pain Type", ChestPainType_encode.classes_)
restingBP = st.slider("Resting BP", 50, 250, 120)
cholesterol = st.slider("Cholesterol", 50, 600, 200)
fastingBS = st.selectbox("Fasting Blood Sugar", [0, 1])
restingECG = st.selectbox("Resting ECG", RestingECG_encode.classes_)
maxhr = st.slider("Max Heart Rate", 60, 250, 150)
exerciseangina = st.selectbox("Exercise Angina", ExerciseAngina_encode.classes_)
oldpeak = st.number_input("Oldpeak", -5.0, 10.0, 1.0)
st_slope = st.selectbox("ST Slope", ST_Slope_encode.classes_)

# ================= ENCODING (IMPORTANT FIX) =================
sex_encoded = int(sex_encode.transform([sex])[0])
chestpain_encoded = int(ChestPainType_encode.transform([chestpain])[0])
restingECG_encoded = int(RestingECG_encode.transform([restingECG])[0])
exerciseangina_encoded = int(ExerciseAngina_encode.transform([exerciseangina])[0])
st_slope_encoded = int(ST_Slope_encode.transform([st_slope])[0])

# ================= DATAFRAME =================
input_data = {
    "Age": [age],
    "Sex": [sex_encoded],
    "ChestPainType": [chestpain_encoded],
    "RestingBP": [restingBP],
    "Cholesterol": [cholesterol],
    "FastingBS": [fastingBS],
    "RestingECG": [restingECG_encoded],
    "MaxHR": [maxhr],
    "ExerciseAngina": [exerciseangina_encoded],
    "Oldpeak": [oldpeak],
    "ST_Slope": [st_slope_encoded]
}

FEATURE_ORDER = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
    "Oldpeak", "ST_Slope"
]

df = pd.DataFrame(input_data)[FEATURE_ORDER]

# ================= PREDICTION =================
if st.button("Predict"):
    scaled_data = scaler.transform(df)
    prediction = model.predict(scaled_data)[0]

    if prediction == 1:
        st.error("⚠️ High risk of Heart Failure")
    else:
        st.success("✅ Low risk of Heart Failure")
