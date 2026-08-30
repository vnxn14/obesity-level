import pandas as pd
from sklearn.neural_network import MLPClassifier

from data_loader import (
    evaluate_classifier,
    load_all_processed_data,
    run_prediction,
)
from model_cache import load_or_train


def train_ann_model():
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, _X = load_all_processed_data()

    ann_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        early_stopping=True,
        random_state=42,
    )
    ann_model.fit(X_train_scaled, y_train)

    accuracy, precision, recall, f1, roc_auc, cm = evaluate_classifier(ann_model, X_test_scaled, y_test)
    return ann_model, scaler, X_train_scaled.columns, accuracy, precision, recall, f1, roc_auc, cm


def get_trained_ann_model():
    return load_or_train("ann", train_ann_model)


def predict_ann(model, scaler, expected_columns_order, single_input_df):
    return run_prediction(model, scaler, expected_columns_order, single_input_df)


if __name__ == "__main__":
    model, scaler, expected_columns_order, accuracy, precision, recall, f1, roc_auc, cm = train_ann_model()
    print(f"ANN trained. Test Accuracy: {accuracy:.2f}% | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | ROC-AUC: {roc_auc:.3f}")
