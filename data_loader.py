import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

@st.cache_data
def load_all_processed_data():

    # --- 1. LOAD DATA ---
    df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

    # --- 2. DEDUPLICATION ---
    df_clean = df.drop_duplicates()

    # --- 3. COLUMN RENAMING & TEXT CLEANING ---
    column_rename_map = {
        'family_history_with_overweight': 'Family_History', 'FAVC': 'High_Caloric_Food',
        'FCVC': 'Veg_Consumption', 'NCP': 'Num_Main_Meals', 'CAEC': 'Food_Between_Meals',
        'CH2O': 'Water_Intake', 'SCC': 'Calorie_Monitoring', 'FAF': 'Physical_Activity',
        'TUE': 'Tech_Usage_Time', 'CALC': 'Alcohol_Intake', 'MTRANS': 'Transport_Mode',
        'NObeyesdad': 'Obesity_Level'
    }
    df_clean = df_clean.rename(columns=column_rename_map)

    text_columns = ['Gender', 'Family_History', 'High_Caloric_Food', 'SMOKE',
                    'Calorie_Monitoring', 'Food_Between_Meals', 'Alcohol_Intake', 'Transport_Mode']
    for col in text_columns:
        if col in df_clean.columns and df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].str.strip().str.lower()

    # --- 4. DATA TYPE CONVERSION & ACCURACY FIXES ---
    df_clean['Age'] = df_clean['Age'].round(0).astype(int)
    df_clean['Height'] = df_clean['Height'].round(2)
    df_clean['Weight'] = df_clean['Weight'].round(1)

    behavioral_cols = ['Veg_Consumption', 'Num_Main_Meals', 'Water_Intake', 'Physical_Activity', 'Tech_Usage_Time']
    df_clean[behavioral_cols] = df_clean[behavioral_cols].round(2)

    # --- 5. FEATURE ENGINEERING ---
    df_clean['BMI'] = (df_clean['Weight'] / (df_clean['Height'] ** 2)).round(2)

    # --- 6. CATEGORICAL ENCODING ---
    # Binary Mappings
    binary_map = {'yes': 1, 'no': 0, 'male': 1, 'female': 0}
    binary_cols = ['Gender', 'Family_History', 'High_Caloric_Food', 'SMOKE', 'Calorie_Monitoring']
    for col in binary_cols:
        df_clean[col] = df_clean[col].map(binary_map).astype(int)

    # Ordinal Mappings
    ordinal_map = {'no': 0, 'sometimes': 1, 'frequently': 2, 'always': 3}
    df_clean['Food_Between_Meals'] = df_clean['Food_Between_Meals'].replace(ordinal_map).astype(int)
    df_clean['Alcohol_Intake'] = df_clean['Alcohol_Intake'].replace(ordinal_map).astype(int)

    # Target Label Encodings
    target_map = {
        'Insufficient_Weight': 0, 
        'Normal_Weight': 1,
        'Overweight_Level_I': 2,
        'Overweight_Level_II': 3,
        'Obesity_Type_I': 4,
        'Obesity_Type_II': 5,
        'Obesity_Type_III': 6
    }
    df_clean['Obesity_Level_Encoded'] = df_clean['Obesity_Level'].map(target_map)

    # Nominal One-Hot Encoding
    if 'Transport_Mode' in df_clean.columns:
        df_clean = pd.get_dummies(df_clean, columns=['Transport_Mode'], prefix='MTRANS', dtype=int)

    # Clean residual booleans to integers
    bool_cols = df_clean.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        df_clean[bool_cols] = df_clean[bool_cols].astype(int)

    # --- 7. FEATURE EXTRACTION & ISOLATION ---
    # In Section 7, explicitly drop BMI along with the targets
    X = df_clean.drop(columns=['Obesity_Level', 'Obesity_Level_Encoded', 'BMI'])
    y = df_clean['Obesity_Level_Encoded']

    # --- 8. TRAIN TEST SPLITTING ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # --- 9. FEATURE SCALING ---
    scaler = StandardScaler()
    scaled_features = [
        'Age', 'Height', 'Weight', 'Veg_Consumption',
        'Num_Main_Meals', 'Water_Intake', 'Physical_Activity',
        'Tech_Usage_Time', 'Food_Between_Meals', 'Alcohol_Intake'
    ]

    # Avoid modifications over slice memory references by copying explicitly
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    # Fit transformations accurately matching cross-validation boundaries
    X_train_scaled[scaled_features] = scaler.fit_transform(X_train[scaled_features])
    X_test_scaled[scaled_features] = scaler.transform(X_test[scaled_features])

    # Return exactly what your pipeline layout needs
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X
