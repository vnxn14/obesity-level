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


