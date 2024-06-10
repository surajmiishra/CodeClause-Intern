# heart_disease_ui.py

import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('heart_disease_model.pkl')

# Title of the app
st.title('Heart Disease Risk Assessment')

# Input fields for user data
age = st.number_input('Age', min_value=0, max_value=120, value=25)
sex = st.selectbox('Sex', [0, 1]) # 0: Female, 1: Male
cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3])
trestbps = st.number_input('Resting Blood Pressure', min_value=80, max_value=200, value=120)
chol = st.number_input('Serum Cholesterol', min_value=100, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
restecg = st.selectbox('Resting Electrocardiographic Results', [0, 1, 2])
thalach = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)
exang = st.selectbox('Exercise Induced Angina', [0, 1])
oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox('Slope of the Peak Exercise ST Segment', [0, 1, 2])
ca = st.number_input('Number of Major Vessels Colored by Fluoroscopy', min_value=0, max_value=4, value=0)
thal = st.selectbox('Thalassemia', [0, 1, 2, 3])

# Prediction button
if st.button('Predict'):
    user_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
    prediction = model.predict(user_data)
    
    if prediction == 1:
        st.warning('High risk of heart disease.')
    else:
        st.success('Low risk of heart disease.')
