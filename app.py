import streamlit as st 
import pickle
import tensorflow
from tensorflow.keras.models import load_model 
import pandas as pd
import numpy as np 


st.title("Heart Failure Prediction")

st.write("Let's check the heart condition of the patience")

st.write("Please fill the requested details of patience's below")

##let's take input of the user


with open("sex_encode.pkl","rb")as f:
     sex_encode=pickle.load(f)

with open("chestpain.pkl","rb")as f:
     ChestPainType_encode=pickle.load(f)
     
with open("RestingECG_encode.pkl","rb")as f:
     RestingECG_encode=pickle.load(f)
     
with open("ExerciseAngina_encode.pkl","rb")as f:
     ExerciseAngina_encode=pickle.load(f)
     
with open("ST_Slope_encode.pkl","rb")as f:
     ST_Slope_encode=pickle.load(f)
     
with open("scaler.pkl","rb")as f:
    scaler=pickle.load(f)
with open("model.pkl","rb")as f:
    model=pickle.load(f)

cols=["Age	Sex		RestingBP	Cholesterol	FastingBS	RestingECG	MaxHR	ExerciseAngina	Oldpeak	ST_Slope"]
age=st.slider("Please enter the patient age: ",0,100,15)
sex=st.selectbox("Please select the patience gender: ",sex_encode.classes_)
chestpaintype=st.selectbox("pain type: ",ChestPainType_encode.classes_)
restingBP=st.slider("BP:",10,1000,100)
cholestrol=st.slider("Cholestrol:",10,1000,100)
FastingBS=st.selectbox("fasting :",[0,1])
restingECG=st.selectbox("ECG: ",RestingECG_encode.classes_)
maxhr=st.slider("Max Hr: ",100,300,5)
exerciseangina=st.selectbox("Exercise: ",ExerciseAngina_encode.classes_)
oldpeak=st.number_input("old peak: ",-100.0,500.0,0.8)
st_slope=st.selectbox("st slope: ",ST_Slope_encode.classes_)

input_data={
    "Age":[age],
    "Sex":[sex_encode.transform([sex])],
    "ChestPainType":[ChestPainType_encode.transform([chestpaintype])],
    "RestingBP":[restingBP],
    "Cholesterol":[cholestrol],
    "FastingBS":[FastingBS],
    "RestingECG":[RestingECG_encode.transform([restingECG])],
    "MaxHR":[maxhr],
    "ExerciseAngina":[ExerciseAngina_encode.transform([exerciseangina])],
    "Oldpeak":[oldpeak],
    "ST_Slope":[ST_Slope_encode.transform([st_slope])]
}

df=pd.DataFrame(input_data)

scaled_data=scaler.transform(df)

if st.button("Predict"):
    pred=model.predict(scaled_data)
    if pred == 1:
        st.write("Patience heart is failure")
    else:
        st.write("Patience heart is good")


