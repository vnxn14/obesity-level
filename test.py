import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# =========================================================================
# 1. LOAD DATASET & DEDUPLICATION
# =========================================================================
# Read raw dataset
df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

# Remove duplicate records
initial_rows = len(df)
df = df.drop_duplicates()
print(f"[INFO] Removed {initial_rows - len(df)} duplicate rows. New row count: {len(df)}")

# Check for missing values
missing_count = df.isnull().sum().sum()
print(f"[INFO] Total missing values found: {missing_count}")

# =========================================================================
# 2. FEATURE RELABELING (HEADER STANDARDIZATION)
# =========================================================================
column_rename_map = {
    'family_history_with_overweight': 'Family_History',
    'FAVC': 'High_Caloric_Food',
    'FCVC': 'Veg_Consumption',
    'NCP': 'Num_Main_Meals',
    'CAEC': 'Food_Between_Meals',
    'CH2O': 'Water_Intake',
    'SCC': 'Calorie_Monitoring',
    'FAF': 'Physical_Activity',
    'TUE': 'Tech_Usage_Time',
    'CALC': 'Alcohol_Intake',
    'MTRANS': 'Transport_Mode',
    'NObeyesdad': 'Obesity_Level'
}

df = df.rename(columns=column_rename_map)

# =========================================================================
# 3. TEXT NORMALIZATION & DECIMAL PRECISION CLEANING
# =========================================================================
# Normalize categorical text features (strip whitespace and convert to lower case)
text_columns = ['Gender', 'Family_History', 'High_Caloric_Food', 'SMOKE',
                'Calorie_Monitoring', 'Food_Between_Meals', 'Alcohol_Intake', 'Transport_Mode']

for col in text_columns:
    if col in df.columns and df[col].dtype == 'object':
        df[col] = df[col].str.strip().str.lower()

# Clean decimal precision introduced by SMOTE generation
df['Age'] = df['Age'].round(0).astype(int)
df['Height'] = df['Height'].round(2)
df['Weight'] = df['Weight'].round(1)

behavioral_cols = ['Veg_Consumption', 'Num_Main_Meals', 'Water_Intake', 'Physical_Activity', 'Tech_Usage_Time']
df[behavioral_cols] = df[behavioral_cols].round(2)

# =========================================================================
# 4. FEATURE ENGINEERING
# =========================================================================
# Engineer Body Mass Index (BMI) feature
df['BMI'] = (df['Weight'] / (df['Height'] ** 2)).round(2)

# =========================================================================
# 5. CATEGORICAL ENCODING
# =========================================================================
# A. Binary Mapping (Safe replacement using .replace to avoid NaNs)
binary_map = {'yes': 1, 'no': 0, 'male': 1, 'female': 0}
binary_cols = ['Gender', 'Family_History', 'High_Caloric_Food', 'SMOKE', 'Calorie_Monitoring']

for col in binary_cols:
    df[col] = df[col].replace(binary_map).astype(int)

# B. Ordinal Mapping
ordinal_map = {'no': 0, 'sometimes': 1, 'frequently': 2, 'always': 3}
df['Food_Between_Meals'] = df['Food_Between_Meals'].replace(ordinal_map).astype(int)
df['Alcohol_Intake'] = df['Alcohol_Intake'].replace(ordinal_map).astype(int)

# C. Target Variable Encoding (0 to 6)
target_map = {
    'Insufficient_Weight': 0,
    'Normal_Weight': 1,
    'Overweight_Level_I': 2,
    'Overweight_Level_II': 3,
    'Obesity_Type_I': 4,
    'Obesity_Type_II': 5,
    'Obesity_Type_III': 6
}
df['Obesity_Level_Encoded'] = df['Obesity_Level'].map(target_map)

# D. One-Hot Encoding for Nominal Category (Transport_Mode)
df = pd.get_dummies(df, columns=['Transport_Mode'], prefix='MTRANS', dtype=int)

# Convert any remaining boolean columns to int
bool_cols = df.select_dtypes(include='bool').columns
if len(bool_cols) > 0:
    df[bool_cols] = df[bool_cols].astype(int)

# =========================================================================
# 6. FEATURE SCALING (FOR k-NN, SVM, AND ANN MODEL PREPARATION)
# =========================================================================
scaler = StandardScaler()
scaled_features = ['Age', 'Height', 'Weight', 'Veg_Consumption',
                   'Num_Main_Meals', 'Water_Intake', 'Physical_Activity', 'Tech_Usage_Time', 'BMI']

# Create a scaled copy of the dataframe for distance-based models
df_scaled = df.copy()
df_scaled[scaled_features] = scaler.fit_transform(df_scaled[scaled_features])

# Save clean dataset
df.to_csv('cleaned_obesity_dataset.csv', index=False)
df_scaled.to_csv('cleaned_obesity_dataset_scaled.csv', index=False)

print(f"\n[SUCCESS] Preprocessing completed! Cleaned dataset shape: {df.shape}")

# =========================================================================
# 7. TRAIN-TEST SPLIT (80% TRAIN, 20% TEST)
# =========================================================================
# Define Features (X) and Target (y) using the SCALED dataframe
X = df_scaled.drop(columns=['Obesity_Level', 'Obesity_Level_Encoded'])
y = df_scaled['Obesity_Level_Encoded']

# Perform stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Combine features and targets back together for saving (Scaled)
train_dataset = X_train.copy()
train_dataset['Obesity_Level_Encoded'] = y_train

test_dataset = X_test.copy()
test_dataset['Obesity_Level_Encoded'] = y_test

# Save scaled files
train_dataset.to_csv('obesity_train_scaled.csv', index=False)
test_dataset.to_csv('obesity_test_scaled.csv', index=False)

# --- NEW: EXTRACTION OF UNSCALED TEST DATA FOR USER TESTING ---
# We grab the exact same rows from the unscaled 'df' using the test split's index
unscaled_test_dataset = df.loc[X_test.index].copy()

# Save the unscaled test data to a CSV file
unscaled_test_dataset.to_csv('obesity_test_unscaled.csv', index=False)

print(f"\n[SUCCESS] Split datasets saved!")
print(f" -> Training file saved as: 'obesity_train_scaled.csv' {train_dataset.shape}")
print(f" -> Testing file saved as : 'obesity_test_scaled.csv' {test_dataset.shape}")
print(f" -> Unscaled User Test file: 'obesity_test_unscaled.csv' {unscaled_test_dataset.shape}")
