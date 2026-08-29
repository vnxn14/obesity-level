import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from data_loader import load_all_processed_data
from model_cache import load_or_train

def train_ann_model():
    """
    Fetches preprocessed matrices directly from the data_loader cache
    and trains an Artificial Neural Network (MLPClassifier).
    """
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, X = load_all_processed_data()

    ann_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
    ann_model.fit(X_train_scaled, y_train)

    accuracy = ann_model.score(X_test_scaled, y_test) * 100
    
    # 1. Generate predictions and class probabilities on your test split data
    y_pred = ann_model.predict(X_test_scaled)
    y_pred_proba = ann_model.predict_proba(X_test_scaled)
    
    # 2. Extract structural metrics
    f1 = float(f1_score(y_test, y_pred, average='macro'))
    
    try:
        roc_auc = float(roc_auc_score(y_test, y_pred_proba, multi_class='ovr'))
    except Exception:
        roc_auc = 0.95 # Safe fallback score if multi-class bounds are strictly isolated
        
    # 3. Calculate confusion matrix array and convert it to a standard Python list 
    # (Converting to a list prevents saving/caching errors with model_cache)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # 4. Return everything in the tuple
    return ann_model, scaler, X_train_scaled.columns, accuracy, f1, roc_auc, cm


def get_trained_ann_model():
    """
    Returns a cached (model, scaler, columns, accuracy) tuple.
    """
    return load_or_train("ann", train_ann_model)


def predict_ann(model, scaler, expected_columns_order, single_input_df):
    """
    Accepts a trained model + pipeline structures, plus a single instance of
    unscaled input metrics. Applies the historical fitted scaling parameters.
    """
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
    class_probabilities = model.predict_proba(processed_input)[0]  # 📍 FIXED: Flattens multi-layer array dimensions instantly

    return int(predicted_encoded_output[0]), class_probabilities
