import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    # =========================================================================
    # 1. LOAD DATASET & DEDUPLICATION
    # =========================================================================
    df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
    df = df.drop_duplicates()
    
    # =========================================================================
    # 2. FEATURE RELABELING
    # =========================================================================
    column_rename_map = {
        'family_history_with_overweight': 'Family_History', 'FAVC': 'High_Caloric_Food',
        'FCVC': 'Veg_Consumption', 'NCP': 'Num_Main_Meals', 'CAEC': 'Food_Between_Meals',
        'CH2O': 'Water_Intake', 'SCC': 'Calorie_Monitoring', 'FAF': 'Physical_Activity',
        'TUE': 'Tech_Usage_Time', 'CALC': 'Alcohol_Intake', 'MTRANS': 'Transport_Mode',
        'NObeyesdad': 'Obesity_Level'
    }
    df = df.rename(columns=column_rename_map)

    # =========================================================================
    # 3. TEXT NORMALIZATION & PRECISION CLEANING
    # =========================================================================
    text_columns = ['Gender', 'Family_History', 'High_Caloric_Food', 'SMOKE',
                    'Calorie_Monitoring', 'Food_Between_Meals', 'Alcohol_Intake', 'Transport_Mode']
    for col in text_columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].str.strip().str.lower()

    df['Age'] = df['Age'].round(0).astype(int)
    df['Height'] = df['Height'].round(2)
    df['Weight'] = df['Weight'].round(1)

    behavioral_cols = ['Veg_Consumption', 'Num_Main_Meals', 'Water_Intake', 'Physical_Activity', 'Tech_Usage_Time']
    df[behavioral_cols] = df[behavioral_cols].round(2)

    # =========================================================================
    # 4. FEATURE ENGINEERING (BMI)
    # =========================================================================
    df['BMI'] = (df['Weight'] / (df['Height'] ** 2)).round(2)

    # =========================================================================
    # 5. CATEGORICAL ENCODING
    # =========================================================================
    binary_map = {'yes': 1, 'no': 0, 'male': 1, 'female': 0}
    binary_cols = ['Gender', 'Family_History', 'High_Caloric_Food', 'SMOKE', 'Calorie_Monitoring']
    for col in binary_cols:
        df[col] = df[col].replace(binary_map).astype(int)

    ordinal_map = {'no': 0, 'sometimes': 1, 'frequently': 2, 'always': 3}
    df['Food_Between_Meals'] = df['Food_Between_Meals'].replace(ordinal_map).astype(int)
    df['Alcohol_Intake'] = df['Alcohol_Intake'].replace(ordinal_map).astype(int)

    target_map = {
        'Insufficient_Weight': 0, 'Normal_Weight': 1, 'Overweight_Level_I': 2,
        'Overweight_Level_II': 3, 'Obesity_Type_I': 4, 'Obesity_Type_II': 5, 'Obesity_Type_III': 6
    }
    df['Obesity_Level_Encoded'] = df['Obesity_Level'].map(target_map)
    df = pd.get_dummies(df, columns=['Transport_Mode'], prefix='MTRANS', dtype=int)

    bool_cols = df.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # =========================================================================
    # 6. FEATURE SCALING
    # =========================================================================
    scaler = StandardScaler()
    scaled_features = ['Age', 'Height', 'Weight', 'Veg_Consumption',
                       'Num_Main_Meals', 'Water_Intake', 'Physical_Activity', 'Tech_Usage_Time', 'BMI']
    
    df_scaled = df.copy()
    df_scaled[scaled_features] = scaler.fit_transform(df_scaled[scaled_features])

    # =========================================================================
    # 7. TRAIN-TEST SPLIT
    # =========================================================================
    X = df_scaled.drop(columns=['Obesity_Level', 'Obesity_Level_Encoded'])
    y = df_scaled['Obesity_Level_Encoded']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Grab unscaled data for user testing using matching test split indices
    df_test_unscaled = df.loc[X_test.index].copy()

    # RETURN all components directly in RAM memory
    return X_train, X_test, y_train, y_test, df_test_unscaled
