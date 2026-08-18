import streamlit as st
import pandas as pd
import numpy as np

from ANN import predict_ann

# Page configurations
st.set_page_config(page_title="Obesity Risk Predictor", layout="wide")
st.title("🏋️‍♂️ Multi-Model Obesity Risk Prediction Dashboard")

# --- 2. SIDEBAR CONFIGURATION ---
st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose a Machine Learning Model",
    ["ANN (Artificial Neural Network)", "KNN (K-Nearest Neighbors)", "SVM (Support Vector Machine)", "Decision Tree"],
    index=0  # Defaults selection directly to ANN
)

st.sidebar.markdown("---")
st.sidebar.info(f"Active Model: **{selected_model}**")

# --- 3. MAIN USER INPUT FORM ---
st.header("👤 Enter Individual Attributes & Habits")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    height = st.number_input("Height (meters)", min_value=1.0, max_value=2.5, value=1.75, step=0.01)
    weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.1)

with col2:
    family_history = st.selectbox("Family History with Overweight?", ["Yes", "No"])
    high_caloric = st.selectbox("Frequently Consume High Caloric Food?", ["Yes", "No"])
    veg_cons = st.slider("Vegetables Consumption Frequency (0-3)", 0.0, 3.0, 2.0, step=0.01)
    main_meals = st.slider("Number of Main Meals Daily (1-4)", 1.0, 4.0, 3.0, step=0.01)
    food_between = st.selectbox("Eat Food Between Meals?", ["No", "Sometimes", "Frequently", "Always"])

with col3:
    water = st.slider("Daily Water Intake (Liters, 1-3)", 1.0, 3.0, 2.0, step=0.01)
    calorie_mon = st.selectbox("Monitor Calories Consumed Daily?", ["Yes", "No"])
    phys_act = st.slider("Physical Activity Frequency (0-3)", 0.0, 3.0, 1.0, step=0.01)
    tech_usage = st.slider("Time Using Tech Devices (0-2 hours)", 0.0, 2.0, 1.0, step=0.01)
    alcohol = st.selectbox("Alcohol Intake Frequency?", ["No", "Sometimes", "Frequently", "Always"])
    transport = st.selectbox("Primary Transportation Mode", ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"])

# --- 4. INPUT DATA PREPROCESSING PIPELINE ---
raw_input_data = {
    'Gender': 1 if gender.lower() == 'male' else 0,
    'Age': int(np.round(age, 0)),
    'Height': round(height, 2),
    'Weight': round(weight, 1),
    'Family_History': 1 if family_history.lower() == 'yes' else 0,
    'High_Caloric_Food': 1 if high_caloric.lower() == 'yes' else 0,
    'Veg_Consumption': round(veg_cons, 2),
    'Num_Main_Meals': round(main_meals, 2),
    'Food_Between_Meals': {'no': 0, 'sometimes': 1, 'frequently': 2, 'always': 3}[food_between.lower()],
    'SMOKE': 0, 
    'Water_Intake': round(water, 2),
    'Calorie_Monitoring': 1 if calorie_mon.lower() == 'yes' else 0,
    'Physical_Activity': round(phys_act, 2),
    'Tech_Usage_Time': round(tech_usage, 2),
    'Alcohol_Intake': {'no': 0, 'sometimes': 1, 'frequently': 2, 'always': 3}[alcohol.lower()],
    'BMI': round(weight / (height ** 2), 2)
}

# Construct matching One-Hot encoded structure for Transport Mode
transport_modes = ['Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking']
for mode in transport_modes:
    feature_name = f"MTRANS_{mode}"
    # Adding .lower() ensures a safe match regardless of UI string casing
    raw_input_data[feature_name] = 1 if transport.lower() == mode.lower() else 0

input_df = pd.DataFrame([raw_input_data])

target_reverse_map = {
    0: 'Insufficient Weight', 1: 'Normal Weight', 2: 'Overweight Level I',
    3: 'Overweight Level II', 4: 'Obesity Type I', 5: 'Obesity Type II', 6: 'Obesity Type III'
}

# --- 5. PREDICTION TRIGGERS ---
st.markdown("---")
if st.button("🚀 Run Obesity Classification Risk Assessment", type="primary"):
    
    if selected_model == "ANN (Artificial Neural Network)":
        with st.spinner("Processing Artificial Neural Network..."):
            try:
                prediction_index, accuracy_score = predict_ann(input_df)
                predicted_label = target_reverse_map.get(prediction_index, "Unknown Class")
                
                st.success("### Prediction Diagnostic Output Completed Successfully!")
                st.metric(label="Predicted Assessment Category", value=predicted_label)
                st.info(f"Model Baseline Benchmark Accuracy: **{accuracy_score:.2f}%**")
            except Exception as e:
                st.error(f"An unexpected computational error occurred: {e}")            
    else:
        st.warning(f"Functional pipelines for **{selected_model}** are currently preparation blueprints. Please select **ANN** to run predictions.")
