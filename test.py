import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
print("Data loaded successfully!")
print("Initial shape of the dataset:", df.shape)

# Mapping dictionary for renaming columns
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

# Rename columns in place
df = df.rename(columns=column_rename_map)

# View updated column names
print(df.columns.tolist())

df = df.drop_duplicates()
print("Duplicates removed. New shape:", df.shape)

# Plot a heatmap of missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Value Heatmap')
plt.show()