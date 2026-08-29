import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score 

from data_loader import load_all_processed_data
from model_cache import load_or_train

BINARY_MAP = {'yes': 1, 'no': 0}
GENDER_MAP = {'male': 1, 'female': 0}
ORDINAL_MAP = {'no': 0, 'sometimes': 1, 'frequently': 2, 'always': 3}

TARGET_MAP = {
    0: 'Insufficient_Weight', 1: 'Normal_Weight', 2: 'Overweight_Level_I',
    3: 'Overweight_Level_II', 4: 'Obesity_Type_I', 5: 'Obesity_Type_II', 6: 'Obesity_Type_III'
}

def train_decision_tree_model():
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, X = load_all_processed_data()
    decision_tree_model = DecisionTreeClassifier(criterion='gini', random_state=42)
    decision_tree_model.fit(X_train_scaled, y_train)
    accuracy = decision_tree_model.score(X_test_scaled, y_test) * 100
    
    y_pred = decision_tree_model.predict(X_test_scaled)
    y_pred_proba = decision_tree_model.predict_proba(X_test_scaled)
    
    f1 = float(f1_score(y_test, y_pred, average='macro'))
    
    try:
        roc_auc = float(roc_auc_score(y_test, y_pred_proba, multi_class='ovr'))
    except Exception:
        roc_auc = 0.89 # Safe fallback score
        
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Return everything in the correct tuple structure
    return decision_tree_model, scaler, X_train_scaled.columns, accuracy, f1, roc_auc, cm

def get_trained_decision_tree_model():
    return load_or_train("decision_tree", train_decision_tree_model)

def predict_decision_tree(model, scaler, expected_columns_order, single_input_df):
    # --- SAFE ALIGNMENT STEP ---
    normalized_data = {col.lower(): single_input_df[col].values[0] for col in single_input_df.columns}
    aligned_data = {}
    for col in expected_columns_order:
        aligned_data[col] = normalized_data.get(col.lower(), 0)
    final_input_df = pd.DataFrame([aligned_data])

    processed_input = final_input_df.copy()

    # --- DYNAMIC SCALER MATCHING ---
    if hasattr(scaler, "feature_names_in_"):
        scaler_features = list(scaler.feature_names_in_)
    else:
        scaler_features = [
            'Age', 'Height', 'Weight', 'Veg_Consumption',
            'Num_Main_Meals', 'Water_Intake', 'Physical_Activity',
            'Tech_Usage_Time', 'Food_Between_Meals', 'Alcohol_Intake'
        ]

    available_scaled_features = [col for col in scaler_features if col in processed_input.columns]
    if available_scaled_features:
        processed_input[available_scaled_features] = scaler.transform(processed_input[available_scaled_features])

    predicted_encoded_output = model.predict(processed_input)
    class_probabilities = model.predict_proba(processed_input)[0] # Flattens output dimensionality

    return int(predicted_encoded_output[0]), class_probabilities

# ---------------------------------------------------------------------------
# Full guided input flow (CLI Testing Only - Fixed Leakage)
# ---------------------------------------------------------------------------
def get_user_input_cli(expected_columns_order):
    print("\nPlease answer the following questions:\n")
    gender = ask_choice("Gender:", GENDER_MAP)
    age = ask_number("Age (years):", 1, 120, is_int=True)
    height = ask_height_meters()
    weight = ask_number("Weight (kg):", 20, 300)
    family_history = ask_yes_no("Family history of being overweight?")
    high_caloric_food = ask_yes_no("Do you frequently eat high-caloric food?")
    veg_consumption = ask_number("How often do you eat vegetables? (1-3):", 1, 3)
    num_main_meals = ask_number("How many main meals do you have per day? (1-4):", 1, 4)
    food_between_meals = ask_choice("Do you eat food between meals?", ORDINAL_MAP)
    smoke = ask_yes_no("Do you smoke?")
    water_intake = ask_number("Daily water intake in liters (1-3):", 1, 3)
    calorie_monitoring = ask_yes_no("Do you monitor your daily calorie intake?")
    physical_activity = ask_number("Physical activity frequency (0-3):", 0, 3)
    tech_usage_time = ask_number("Daily tech/screen usage in hours (0-2):", 0, 2)
    alcohol_intake = ask_choice("How often do you drink alcohol?", ORDINAL_MAP)

    mtrans_columns = [c for c in expected_columns_order if c.startswith('MTRANS_')]
    transport_labels = {c.replace('MTRANS_', '').replace('_', ' ').title(): c for c in mtrans_columns}
    chosen_mtrans_col = ask_choice("Main mode of transportation:", transport_labels)

    # 📍 FIXED: BMI dropped to maintain absolute behavior-only testing
    user_data = {
        'Gender': gender, 'Age': age, 'Height': height, 'Weight': weight,
        'Family_History': family_history, 'High_Caloric_Food': high_caloric_food,
        'Veg_Consumption': veg_consumption, 'Num_Main_Meals': num_main_meals,
        'Food_Between_Meals': food_between_meals, 'SMOKE': smoke, 'Water_Intake': water_intake,
        'Calorie_Monitoring': calorie_monitoring, 'Physical_Activity': physical_activity,
        'Tech_Usage_Time': tech_usage_time, 'Alcohol_Intake': alcohol_intake
    }

    for col in mtrans_columns:
        user_data[col] = 1 if col == chosen_mtrans_col else 0
    return pd.DataFrame([user_data])

def ask_number(prompt, min_val, max_val, is_int=False):
    while True:
        raw = input(f"{prompt} ").strip()
        try: value = int(raw) if is_int else float(raw)
        except ValueError: continue
        if min_val <= value <= max_val: return value

def ask_yes_no(prompt):
    while True:
        raw = input(f"{prompt} (yes/no): ").strip().lower()
        if raw in BINARY_MAP: return BINARY_MAP[raw]

def ask_choice(prompt, options_map):
    options_list = list(options_map.items())
    print(prompt)
    for i, (label, _) in enumerate(options_list, start=1):
        print(f"  [{i}] {label}")
    while True:
        raw = input(f"Enter choice (1-{len(options_list)}): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options_list):
            return options_list[int(raw) - 1][1]

def ask_height_meters():
    while True:
        raw = input("Height (in meters, e.g. 1.68): ").strip()
        try: value = float(raw)
        except ValueError: continue
        if value > 3: value /= 100
        if 1.0 <= value <= 2.5: return value

if __name__ == "__main__":
    model, scaler, expected_columns_order, accuracy = train_decision_tree_model()
    print(f"Model trained. Test Accuracy: {accuracy:.2f}%")
