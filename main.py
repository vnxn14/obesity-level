import streamlit as st
import pickle
import numpy as np

# 1. Load your pre-trained assets
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Obesity Risk Predictor")

# 2. Collect user input
height = st.number_input("Enter Height (meters)", min_value=1.0, max_value=2.5, value=1.7)
weight = st.number_input("Enter Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)

if st.button("Predict"):
    # 3. Format and scale the single input exactly like the training data
    user_data = np.array([[height, weight]])
    user_data_scaled = scaler.transform(user_data)
    
    # 4. Predict
    prediction = model.predict(user_data_scaled)
    st.write(f"Prediction: {prediction[0]}")
