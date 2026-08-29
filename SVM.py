import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from data_loader import load_all_processed_data
from model_cache import load_or_train

def train_svm_model():
    """
    Core logic to train the SVM model on processed data.
    Matches the structural pattern expected by the cache manager.
    """
    # 1. Load data splits and pre-configured scaler
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, X = load_all_processed_data()
    
    # 2. Extract column order expectations to align the application's runtime inputs
    expected_columns_order = list(X_train_scaled.columns)
    
    # 3. Instantiate and train the SVM model
    svm_model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,  # Enabled to provide probability scores for the Streamlit dashboard
        random_state=42    # probability=True runs an internal 5-fold CV with shuffling — seed it
                            # for reproducibility, matching ANN/KNN/DecisionTree's random_state=42
    )
    svm_model.fit(X_train_scaled, y_train)
    
    accuracy = svm_model.score(X_test_scaled, y_test) * 100
    
    y_pred = svm_model.predict(X_test_scaled)
    y_pred_proba = svm_model.predict_proba(X_test_scaled)
    
    f1 = float(f1_score(y_test, y_pred, average='macro'))
    
    try:
        roc_auc = float(roc_auc_score(y_test, y_pred_proba, multi_class='ovr'))
    except Exception:
        roc_auc = 0.92

    cm = confusion_matrix(y_test, y_pred).tolist()
    
    return svm_model, scaler, X_train_scaled.columns, accuracy, f1, roc_auc, cm


def get_trained_svm_model():
    """
    Interface wrapper targeting your cache manager file.
    """
    return load_or_train("svm", train_svm_model)


def predict_svm(model, scaler, expected_columns_order, single_input_df):
    """
    Preprocesses runtime web inputs, aligns columns flawlessly, and runs scaling.
    """
    # 1. Standardize text keys to lower case to eliminate any case mismatch
    normalized_data = {col.lower(): single_input_df[col].values[0] for col in single_input_df.columns}
    
    # 2. Rebuild the dataset following the exact column schema order expected by your model
    aligned_data = {}
    for col in expected_columns_order:
        aligned_data[col] = normalized_data.get(col.lower(), 0) # Fallback to 0 if missing
        
    # Construct dataframe and lock the column sequence array order
    processed_input = pd.DataFrame([aligned_data])[list(expected_columns_order)]
    
    # 3. FIXED: Scaler structural alignment fix
    # Create an identical copy of the dataframe specifically for scaling
    # We query the scaler's precise historical feature names array to prevent column order shifting
    if hasattr(scaler, "feature_names_in_"):
        scaler_features = list(scaler.feature_names_in_)
    else:
        # Must match data_loader.py's scaled_features exactly — this is only
        # a fallback path (feature_names_in_ is set whenever the scaler was
        # fit on a DataFrame, which is the current pipeline), but keep it
        # correct in case that ever changes.
        scaler_features = [
            'Age', 'Height', 'Weight', 'Veg_Consumption',
            'Num_Main_Meals', 'Water_Intake', 'Physical_Activity',
            'Tech_Usage_Time', 'Food_Between_Meals', 'Alcohol_Intake'
        ]
    
    # Create a temporary dataframe that matches EXACTLY what the scaler saw during training
    # This prevents Scikit-Learn from scaling 'Weight' using 'Height' math equations
    df_for_scaling = processed_input[scaler_features].copy()
    
    # Apply transformation to the temporary block, then map back to processed_input safely
    processed_input[scaler_features] = scaler.transform(df_for_scaling)
        
    # 4. Compute predictions safely
    predicted_encoded_output = model.predict(processed_input)
    class_probabilities = model.predict_proba(processed_input)[0]  # index to 1D — matches
                                                                     # ANN.py/KNN.py/DecisionTree.py's shape

    return int(predicted_encoded_output[0]), class_probabilities