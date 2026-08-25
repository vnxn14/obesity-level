import pandas as pd
from sklearn.neural_network import MLPClassifier

from data_loader import load_all_processed_data
from model_cache import load_or_train


def train_ann_model():
    """
    Fetches preprocessed matrices directly from the data_loader cache
    and trains an Artificial Neural Network (MLPClassifier).
    """
    # Unpack everything instantly from memory (prevents duplicate CSV reading)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, X = load_all_processed_data()

    # Train ANN Model (Multi-Layer Perceptron)
    # Using 2 hidden layers (64 neurons and 32 neurons) with adam optimizer
    ann_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
    ann_model.fit(X_train_scaled, y_train)

    # Evaluate performance
    accuracy = ann_model.score(X_test_scaled, y_test) * 100

    # Return model runtime requirements back to the dashboard loop
    return ann_model, scaler, X_train_scaled.columns, accuracy


def get_trained_ann_model():
    """
    Returns a cached (model, scaler, columns, accuracy) tuple. Loads it
    from disk via joblib if a cached version already exists instead of
    retraining — see model_cache.CACHE_VERSION to force a fresh retrain.
    """
    return load_or_train("ann", train_ann_model)


def predict_ann(model, scaler, expected_columns_order, single_input_df):
    """
    Accepts a trained model + pipeline structures, plus a single instance of
    unscaled input metrics. Applies the historical fitted scaling parameters
    and returns the classification integer along with the model's per-class
    probability distribution.
    """
    # --- SAFE ALIGNMENT STEP ---
    # Creates a case-insensitive dictionary map of your input data columns
    normalized_data = {col.lower(): single_input_df[col].values[0] for col in single_input_df.columns}

    # Rebuild input dataframe using the EXACT cased sequence expected by the model
    aligned_data = {}
    for col in expected_columns_order:
        aligned_data[col] = normalized_data.get(col.lower(), 0)  # Fallback to 0 if a key is missing

    final_input_df = pd.DataFrame([aligned_data])
    # ---------------------------

    # Scale user input numbers using the original dataset scale weights
    scaled_features = ['Age', 'Height', 'Weight', 'Veg_Consumption', 'Num_Main_Meals', 'Water_Intake', 'Physical_Activity', 'Tech_Usage_Time', 'BMI']
    processed_input = final_input_df.copy()
    processed_input[scaled_features] = scaler.transform(final_input_df[scaled_features])

    # Process prediction + full probability distribution across all classes
    predicted_encoded_output = model.predict(processed_input)
    class_probabilities = model.predict_proba(processed_input)[0]  # shape (n_classes,), order matches model.classes_

    return int(predicted_encoded_output[0]), class_probabilities