from sklearn.svm import SVC

from data_loader import (
    evaluate_classifier,
    load_all_processed_data,
    run_prediction,
)
from model_cache import load_or_train


def train_svm_model():
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, _X = load_all_processed_data()

    trained_column_order = X_train_scaled.columns

    svm_model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True, 
        random_state=42,
    )
    svm_model.fit(X_train_scaled, y_train)

    accuracy, precision, recall, f1, roc_auc, cm = evaluate_classifier(svm_model, X_test_scaled, y_test)
    return svm_model, scaler, trained_column_order, accuracy, precision, recall, f1, roc_auc, cm


def get_trained_svm_model():
    return load_or_train("svm", train_svm_model)


def predict_svm(model, scaler, expected_columns_order, single_input_df):
    return run_prediction(model, scaler, expected_columns_order, single_input_df)


if __name__ == "__main__":
    model, scaler, expected_columns_order, accuracy, precision, recall, f1, roc_auc, cm = train_svm_model()
    print(f"SVM trained. Test Accuracy: {accuracy:.2f}% | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | ROC-AUC: {roc_auc:.3f}")
