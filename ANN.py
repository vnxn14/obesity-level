import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from sklearn.metrics import classification_report

# DIRECTLY CALL THE LOADER FILE AND IMPORT THE PYTHON FUNCTION
from data_loader import load_and_preprocess_data

print("[INFO] Fetching data directly from data_loader.py function...")
X_train, X_test, y_train, y_test, df_test_unscaled = load_and_preprocess_data()

# Convert labels to One-Hot Encoding for the Neural Network
num_classes = 7
y_train_encoded = to_categorical(y_train, num_classes=num_classes)
y_test_encoded = to_categorical(y_test, num_classes=num_classes)

# =========================================================================
# MODEL TRAINING & EVALUATION (Same as before but running directly from variables)
# =========================================================================
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\n[INFO] Training ANN...")
model.fit(X_train, y_train_encoded, validation_split=0.1, epochs=50, batch_size=32, verbose=1)

print("\n[INFO] Generating Evaluation Metrics...")
y_pred_probs = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
print(classification_report(y_test, y_pred_classes))

# =========================================================================
# USER TESTING DEMO (Using the unscaled data directly from the loader)
# =========================================================================
print("\n[INFO] Simulating User Testing...")
inverse_target_map = {
    0: 'Insufficient_Weight', 1: 'Normal_Weight', 2: 'Overweight_Level_I',
    3: 'Overweight_Level_II', 4: 'Obesity_Type_I', 5: 'Obesity_Type_II', 6: 'Obesity_Type_III'
}

# Grab the first raw record from our memory object
single_user_raw = df_test_unscaled.iloc[0]
print(f" -> Mock Form Input Summary: Age {single_user_raw['Age']}, Weight {single_user_raw['Weight']}kg")
print(f" -> Real Medical Label: {single_user_raw['Obesity_Level']}")

# Extract corresponding scaled feature vectors from X_test 
single_user_scaled = X_test.iloc[[0]]

# Predict
user_pred = np.argmax(model.predict(single_user_scaled, verbose=0), axis=1)
print(f" -> ANN Predicted Label: {inverse_target_map[user_pred[0]]}")
