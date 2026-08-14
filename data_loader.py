import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- 1. LOAD DATA & INITIAL EXPLORATION ---
df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

print("--- Initial Data Check ---")
print(f"Dataset Shape: {df.shape}\n")
print("First 5 Rows:")
print(df.head(), "\n")

# Missing Value Visualisation
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Value Heatmap')
plt.show()


# --- 2. DEDUPLICATION ---
df_clean = df.drop_duplicates()

# Calculate counts directly from shapes
initial_rows, initial_cols = df.shape
post_dup_rows, post_dup_cols = df_clean.shape
duplicate_count = initial_rows - post_dup_rows

print("--- Deduplication Audit ---")
print(f"Original shape: {df.shape} | Cleaned shape: {df_clean.shape}")
print(f"Dropped {duplicate_count} duplicate row(s).\n")

# GRAPH B: Deduplication Row Counts Comparison
plt.figure(figsize=(6, 4))
bars = plt.bar(
    ['Original Raw Rows', 'After Removing Duplicates'], 
    [df.shape[0], df_clean.shape[0]], 
    color=['#1f77b4', '#2ca02c'], 
    width=0.5
)
plt.ylabel("Number of Records")
plt.title(f"Deduplication Audit (Dropped {duplicate_count:,} Rows)")
plt.ylim(0, df.shape[0] * 1.12)
plt.bar_label(bars, fmt='{:,.0f}', padding=3, fontweight='bold')
plt.tight_layout()
plt.show()


# --- 3. COLUMN RENAMING & TEXT CLEANING ---
column_rename_map = {
    'family_history_with_overweight': 'Family_History', 'FAVC': 'High_Caloric_Food',
    'FCVC': 'Veg_Consumption', 'NCP': 'Num_Main_Meals', 'CAEC': 'Food_Between_Meals',
    'CH2O': 'Water_Intake', 'SCC': 'Calorie_Monitoring', 'FAF': 'Physical_Activity',
    'TUE': 'Tech_Usage_Time', 'CALC': 'Alcohol_Intake', 'MTRANS': 'Transport_Mode',
    'NObeyesdad': 'Obesity_Level'
}
df_clean = df_clean.rename(columns=column_rename_map)

print("--- Renamed Columns ---")
print(df_clean.columns.tolist(), "\n")

# Lowercase and strip string features
text_columns = ['Gender', 'Family_History', 'High_Caloric_Food', 'SMOKE',
                'Calorie_Monitoring', 'Food_Between_Meals', 'Alcohol_Intake', 'Transport_Mode']
for col in text_columns:
    if col in df_clean.columns and df_clean[col].dtype == 'object':
        df_clean[col] = df_clean[col].str.strip().str.lower()


# --- 4. DATA TYPE CONVERSION & FEATURE ENGINEERING ---
df_clean['Age'] = df_clean['Age'].round(0).astype(int)
df_clean['Height'] = df_clean['Height'].round(2)
df_clean['Weight'] = df_clean['Weight'].round(1)

behavioral_cols = ['Veg_Consumption', 'Num_Main_Meals', 'Water_Intake', 'Physical_Activity', 'Tech_Usage_Time']
df_clean[behavioral_cols] = df_clean[behavioral_cols].round(2)

# Feature Engineering: BMI Calculation
df_clean['BMI'] = (df_clean['Weight'] / (df_clean['Height'] ** 2)).round(2)

# Save unscaled copy of weight for comparison graphics later
raw_weight_snapshot = df_clean['Weight'].copy()


# --- 5. CATEGORICAL ENCODING ---
# Binary Encoding
binary_map = {'yes': 1, 'no': 0, 'male': 1, 'female': 0}
binary_cols = ['Gender', 'Family_History', 'High_Caloric_Food', 'SMOKE', 'Calorie_Monitoring']
# Using map instead of replace avoids the warning entirely
for col in binary_cols:
    df_clean[col] = df_clean[col].map(binary_map).astype(int)

# Ordinal Encoding
ordinal_map = {'no': 0, 'sometimes': 1, 'frequently': 2, 'always': 3}
df_clean['Food_Between_Meals'] = df_clean['Food_Between_Meals'].replace(ordinal_map).astype(int)
df_clean['Alcohol_Intake'] = df_clean['Alcohol_Intake'].replace(ordinal_map).astype(int)

# Target Map Encoding
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

# Nominal Categorical Encoding (One-Hot Encoding)
if 'Transport_Mode' in df_clean.columns:
    df_clean = pd.get_dummies(df_clean, columns=['Transport_Mode'], prefix='MTRANS', dtype=int)

# Convert any residual boolean columns to integer flag representations
bool_cols = df_clean.select_dtypes(include='bool').columns
if len(bool_cols) > 0:
    df_clean[bool_cols] = df_clean[bool_cols].astype(int)


# --- 6. FEATURE SCALING & VERIFICATION ---
scaler = StandardScaler()
scaled_features = [
    'Age', 'Height', 'Weight', 'Veg_Consumption',
    'Num_Main_Meals', 'Water_Intake', 'Physical_Activity', 
    'Tech_Usage_Time', 'BMI'
]

df_scaled = df_clean.copy()
df_scaled[scaled_features] = scaler.fit_transform(df_scaled[scaled_features])

# GRAPH C: Distribution Plot Before vs. After Scaling
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left Plot: Raw values
sns.histplot(raw_weight_snapshot, kde=True, ax=axes[0], color='blue')
axes[0].set_title(
    f"Original Weight Distribution\nMean: {raw_weight_snapshot.mean():.1f} | Std: {raw_weight_snapshot.std():.1f}"
)
axes[0].set_xlabel("Weight (Kilograms)")

# Right Plot: Standardized values
sns.histplot(df_scaled['Weight'], kde=True, ax=axes[1], color='purple')
axes[1].set_title(
    f"Standardized Weight Distribution\nMean: {df_scaled['Weight'].mean():.1f} | Std: {df_scaled['Weight'].std():.1f}"
)
axes[1].set_xlabel("Weight (Z-Scores / Scaled Values)")

plt.suptitle("Feature Scaling Verification (Distribution Structure Check)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# --- 7. DATA SPLITTING ---
X = df_scaled.drop(columns=['Obesity_Level', 'Obesity_Level_Encoded'])
y = df_scaled['Obesity_Level_Encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
    
df_test_unscaled = df_clean.loc[X_test.index].copy()

print("--- Split Results ---")
print(f" -> Train Features shape: {X_train.shape} | Test Features shape: {X_test.shape}")
print(f" -> Unscaled Test DataFrame preserved with shape: {df_test_unscaled.shape}")
