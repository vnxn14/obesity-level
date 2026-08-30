import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_loader import visualize_dataset_analysis

df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

# missing values and duplicates checking, before and after feature scaling graphs
visualize_dataset_analysis()

# ==========================================
# PLOT 1: Feature Correlation Heatmap
# ==========================================
plt.figure(figsize=(10, 8))  # 1. Open a clean figure
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close() 

# ==========================================
# PLOT 2: Distribution of Obesity Levels
# ==========================================
plt.figure(figsize=(10, 5))  # 3. Open a brand new clean figure
# Assuming 'NObeyesdad' is your target obesity column
sns.countplot(data=df, y="NObeyesdad", order=df["NObeyesdad"].value_counts().index)
plt.title("Distribution of Obesity Levels")
plt.xlabel("Count")
plt.ylabel("Obesity Level")
plt.tight_layout()
plt.savefig("obesity_distribution.png")
plt.close()

