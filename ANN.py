import pandas as pd
from sklearn.neural_network import MLPClassifier

from data_loader import load_all_processed_data

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

def predict_ann(single_input_df):
    """
    Accepts single instance unscaled input metrics, applies historical 
    fitted scaling parameters, and returns the classification integer.
    """
    # Fetch trained model and pipeline structures
    model, scaler, expected_columns_order, accuracy = train_ann_model()
    
    # Force identical feature column arrangement matching baseline limits
    single_input_df = single_input_df[expected_columns_order]
    
    # Scale user input numbers using the original dataset scale weights
    scaled_features = ['Age', 'Height', 'Weight', 'Veg_Consumption', 'Num_Main_Meals', 'Water_Intake', 'Physical_Activity', 'Tech_Usage_Time', 'BMI']
    processed_input = single_input_df.copy()
    processed_input[scaled_features] = scaler.transform(single_input_df[scaled_features])
    
    # Process prediction
    predicted_encoded_output = model.predict(processed_input)
    
    return int(predicted_encoded_output), accuracy
