import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from data_loader import load_all_processed_data


# ---------------------------------------------------------------------------
# Mappings — must mirror data_loader.py exactly, or predictions will be wrong
# ---------------------------------------------------------------------------
BINARY_MAP = {'yes': 1, 'no': 0}
GENDER_MAP = {'male': 1, 'female': 0}
ORDINAL_MAP = {'no': 0, 'sometimes': 1, 'frequently': 2, 'always': 3}

TARGET_MAP = {
    0: 'Insufficient_Weight',
    1: 'Normal_Weight',
    2: 'Overweight_Level_I',
    3: 'Overweight_Level_II',
    4: 'Obesity_Type_I',
    5: 'Obesity_Type_II',
    6: 'Obesity_Type_III',
}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_knn_model():
    """
    Fetches preprocessed matrices directly from the data_loader cache
    and trains a K-Nearest Neighbors classifier.
    """
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, X = load_all_processed_data()

    knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='minkowski', p=2)
    knn_model.fit(X_train_scaled, y_train)

    accuracy = knn_model.score(X_test_scaled, y_test) * 100

    return knn_model, scaler, X_train_scaled.columns, accuracy


def predict_knn(model, scaler, expected_columns_order, single_input_df):
    """
    Accepts a single-row DataFrame of already-encoded (but unscaled) inputs,
    applies the fitted scaler, and returns the predicted class (int) along
    with the model's per-class probability distribution (based on neighbor
    vote weighting, since KNN uses weights='distance').
    """
    # --- SAFE ALIGNMENT STEP (mirrors ANN.py's predict_ann) ---
    # Column names built upstream (e.g. in main.py) may not match the exact
    # casing data_loader.py used when training (e.g. "MTRANS_Automobile" vs
    # "MTRANS_automobile"). Match case-insensitively and rebuild in the
    # exact cased order the trained model expects.
    normalized_data = {col.lower(): single_input_df[col].values[0] for col in single_input_df.columns}
    aligned_data = {}
    for col in expected_columns_order:
        aligned_data[col] = normalized_data.get(col.lower(), 0)  # fallback to 0 if a key is missing
    single_input_df = pd.DataFrame([aligned_data])
    # -----------------------------------------------------------

    scaled_features = ['Age', 'Height', 'Weight', 'Veg_Consumption', 'Num_Main_Meals',
                        'Water_Intake', 'Physical_Activity', 'Tech_Usage_Time', 'BMI']
    processed_input = single_input_df.copy()
    processed_input[scaled_features] = scaler.transform(single_input_df[scaled_features])

    predicted_encoded_output = model.predict(processed_input)
    class_probabilities = model.predict_proba(processed_input)[0]  # order matches model.classes_
    return int(predicted_encoded_output[0]), class_probabilities


# ---------------------------------------------------------------------------
# Input helpers — keep prompts + validation in one place
# ---------------------------------------------------------------------------
def ask_number(prompt, min_val, max_val, is_int=False):
    while True:
        raw = input(f"{prompt} ").strip()
        try:
            value = int(raw) if is_int else float(raw)
        except ValueError:
            print(f"  Please enter a number.")
            continue
        if not (min_val <= value <= max_val):
            print(f"  Please enter a value between {min_val} and {max_val}.")
            continue
        return value


def ask_yes_no(prompt):
    while True:
        raw = input(f"{prompt} (yes/no): ").strip().lower()
        if raw in BINARY_MAP:
            return BINARY_MAP[raw]
        print("  Please answer 'yes' or 'no'.")


def ask_choice(prompt, options_map):
    """options_map: dict of {display_text: encoded_value}"""
    options_list = list(options_map.items())
    print(prompt)
    for i, (label, _) in enumerate(options_list, start=1):
        print(f"  [{i}] {label}")
    while True:
        raw = input(f"Enter choice (1-{len(options_list)}): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options_list):
            return options_list[int(raw) - 1][1]
        print(f"  Please enter a number between 1 and {len(options_list)}.")


def ask_height_meters():
    """Height must be in meters (e.g. 1.68) to match BMI = weight / height^2."""
    while True:
        raw = input("Height (in meters, e.g. 1.68): ").strip()
        try:
            value = float(raw)
        except ValueError:
            print("  Please enter a number.")
            continue
        if value > 3:
            # user probably typed cm (e.g. 168) — auto-correct
            value = value / 100
            print(f"  (Interpreted as {value:.2f} m)")
        if 1.0 <= value <= 2.5:
            return value
        print("  Please enter a realistic height in meters (e.g. 1.0 - 2.5).")


# ---------------------------------------------------------------------------
# Full guided input flow
# ---------------------------------------------------------------------------
def get_user_input_cli(expected_columns_order):
    print("\nPlease answer the following questions:\n")

    gender = ask_choice("Gender:", GENDER_MAP)
    age = ask_number("Age (years):", 1, 120, is_int=True)
    height = ask_height_meters()
    weight = ask_number("Weight (kg):", 20, 300)

    family_history = ask_yes_no("Family history of being overweight?")
    high_caloric_food = ask_yes_no("Do you frequently eat high-caloric food?")
    veg_consumption = ask_number(
        "How often do you eat vegetables? (1 = rarely, 2 = sometimes, 3 = every meal):", 1, 3)
    num_main_meals = ask_number("How many main meals do you have per day? (1-4):", 1, 4)
    food_between_meals = ask_choice("Do you eat food between meals?", ORDINAL_MAP)
    smoke = ask_yes_no("Do you smoke?")
    water_intake = ask_number("Daily water intake in liters (1-3):", 1, 3)
    calorie_monitoring = ask_yes_no("Do you monitor your daily calorie intake?")
    physical_activity = ask_number("Physical activity frequency (0 = none - 3 = high):", 0, 3)
    tech_usage_time = ask_number("Daily tech/screen usage in hours (0 = low - 2 = high):", 0, 2)
    alcohol_intake = ask_choice("How often do you drink alcohol?", ORDINAL_MAP)

    # Transport mode is one-hot encoded — build choices from whatever
    # MTRANS_ columns the trained model actually expects.
    mtrans_columns = [c for c in expected_columns_order if c.startswith('MTRANS_')]
    transport_labels = {c.replace('MTRANS_', '').replace('_', ' ').title(): c for c in mtrans_columns}
    chosen_mtrans_col = ask_choice("Main mode of transportation:", transport_labels)

    bmi = round(weight / (height ** 2), 2)

    user_data = {
        'Gender': gender,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'Family_History': family_history,
        'High_Caloric_Food': high_caloric_food,
        'Veg_Consumption': veg_consumption,
        'Num_Main_Meals': num_main_meals,
        'Food_Between_Meals': food_between_meals,
        'SMOKE': smoke,
        'Water_Intake': water_intake,
        'Calorie_Monitoring': calorie_monitoring,
        'Physical_Activity': physical_activity,
        'Tech_Usage_Time': tech_usage_time,
        'Alcohol_Intake': alcohol_intake,
        'BMI': bmi,
    }

    # Set every MTRANS_ column to 0, then flip the chosen one to 1
    for col in mtrans_columns:
        user_data[col] = 1 if col == chosen_mtrans_col else 0

    return pd.DataFrame([user_data])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model, scaler, expected_columns_order, accuracy = train_knn_model()
    print(f"Model trained. Test Accuracy: {accuracy:.2f}%")

    single_input_df = get_user_input_cli(expected_columns_order)

    prediction, _probabilities = predict_knn(model, scaler, expected_columns_order, single_input_df)
    prediction_label = TARGET_MAP.get(prediction, "Unknown")

    print(f"\nPredicted Obesity Level: {prediction_label}")
    print(f"Model Accuracy: {accuracy:.2f}%")