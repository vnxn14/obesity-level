import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from data_loader import (
    GENDER_MAPPING,
    HABIT_MAPPING,
    evaluate_classifier,
    load_all_processed_data,
    run_prediction,
    target_mapping,
)
from model_cache import load_or_train

BINARY_MAP = {"yes": "yes", "no": "no"}
TARGET_MAP = {code: name for name, code in target_mapping.items()}


def train_knn_model():
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, _X = load_all_processed_data()

    # Capture the exact structural column index directly from X_train_scaled
    trained_column_order = X_train_scaled.columns

    knn_model = KNeighborsClassifier(n_neighbors=5, weights="distance", metric="minkowski", p=2)
    knn_model.fit(X_train_scaled, y_train)

    accuracy, f1, roc_auc, cm = evaluate_classifier(knn_model, X_test_scaled, y_test)
    return knn_model, scaler, trained_column_order, accuracy, f1, roc_auc, cm


def get_trained_knn_model():
    return load_or_train("knn", train_knn_model)


def predict_knn(model, scaler, expected_columns_order, single_input_df):
    return run_prediction(model, scaler, expected_columns_order, single_input_df)


def get_user_input_cli(_expected_columns_order):
    print("\nPlease answer the following questions:\n")

    # Pass lists of raw string keys to preserve feature loader operations
    gender = ask_choice("Gender:", list(GENDER_MAPPING.keys()))
    age = ask_number("Age (years):", 1, 120, is_int=False)
    height = ask_height_meters()
    weight = ask_number("Weight (kg):", 20, 300)
    family_history = ask_yes_no("Family history of being overweight?")
    high_caloric_food = ask_yes_no("Do you frequently eat high-caloric food?")
    veg_consumption = ask_number("How often do you eat vegetables? (1-3):", 1, 3)
    num_main_meals = ask_number("How many main meals do you have per day? (1-4):", 1, 4)
    food_between_meals = ask_choice("Do you eat food between meals?", list(HABIT_MAPPING.keys()))
    smoke = ask_yes_no("Do you smoke?")
    water_intake = ask_number("Daily water intake in liters (1-3):", 1, 3)
    calorie_monitoring = ask_yes_no("Do you monitor your daily calorie intake?")
    physical_activity = ask_number("Physical activity frequency (0-3):", 0, 3)
    tech_usage_time = ask_number("Daily tech/screen usage in hours (0-2):", 0, 2)
    alcohol_intake = ask_choice("How often do you drink alcohol?", list(HABIT_MAPPING.keys()))
    
    # Fixed underscore value mismatch to match raw dataset tracking features
    transport = ask_choice(
        "Main mode of transportation:",
        ["Automobile", "Bike", "Motorbike", "Public Transportation", "Walking"],
    )

    return pd.DataFrame(
        [
            {
                "Gender": gender,
                "Age": age,
                "Height": height,
                "Weight": weight,
                "Family_History_Overweight": family_history,
                "High_Caloric_Food_Freq": high_caloric_food,
                "Vegetable_Consumption_Freq": veg_consumption,
                "Main_Meals_Per_Day": num_main_meals,
                "Food_Between_Meals": food_between_meals,
                "SMOKE": smoke,
                "Water_Intake_Daily": water_intake,
                "Calories_Monitoring": calorie_monitoring,
                "Physical_Activity_Freq": physical_activity,
                "Tech_Device_Usage_Time": tech_usage_time,
                "Alcohol_Consumption": alcohol_intake,
                "Transportation_Method": transport,
            }
        ]
    )


def ask_number(prompt, min_val, max_val, is_int=False):
    while True:
        raw = input(f"{prompt} ").strip()
        try:
            value = int(raw) if is_int else float(raw)
        except ValueError:
            continue
        if min_val <= value <= max_val:
            return value


def ask_yes_no(prompt):
    while True:
        raw = input(f"{prompt} (yes/no): ").strip().lower()
        if raw in BINARY_MAP:
            return BINARY_MAP[raw]


def ask_choice(prompt, options_list):
    print(prompt)
    for i, label in enumerate(options_list, start=1):
        print(f"  [{i}] {label}")
    while True:
        raw = input(f"Enter choice (1-{len(options_list)}): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options_list):
            return options_list[int(raw) - 1]


def ask_height_meters():
    while True:
        raw = input("Height (in meters, e.g. 1.68): ").strip()
        try:
            value = float(raw)
        except ValueError:
            continue
        if value > 3:
            value /= 100
        if 1.0 <= value <= 2.5:
            return value


if __name__ == "__main__":
    model, scaler, expected_columns_order, accuracy, f1, roc_auc, cm = train_knn_model()
    print(f"KNN trained. Test Accuracy: {accuracy:.2f}%  F1: {f1:.3f}  ROC-AUC: {roc_auc:.3f}")
    single_input_df = get_user_input_cli(expected_columns_order)
    prediction, _probabilities = predict_knn(model, scaler, expected_columns_order, single_input_df)
    print(f"\nPredicted Obesity Level: {TARGET_MAP.get(prediction, 'Unknown')}")
