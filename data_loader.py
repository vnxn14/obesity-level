import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Expose target_mapping universally so main.py can use it for labels
target_mapping = {
    "Insufficient_Weight": 0,
    "Normal_Weight": 1,
    "Overweight_Level_I": 2,
    "Overweight_Level_II": 3,
    "Obesity_Type_I": 4,
    "Obesity_Type_II": 5,
    "Obesity_Type_III": 6,
}

CLASS_LABELS = list(range(len(target_mapping)))

GENDER_MAPPING = {"Female": 0, "Male": 1}
HABIT_MAPPING = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}

CATEGORICAL_COLS = [
    "Family_History_Overweight",
    "High_Caloric_Food_Freq",
    "Calories_Monitoring",
    "Transportation_Method",
    "SMOKE",
]

NUMERICAL_COLS = [
    "Age",
    "Height",
    "Weight",
    "Vegetable_Consumption_Freq",
    "Main_Meals_Per_Day",
    "Water_Intake_Daily",
    "Physical_Activity_Freq",
    "Tech_Device_Usage_Time",
    "Alcohol_Consumption",
    "Food_Between_Meals",
]

YES_NO_COLS = [
    "Family_History_Overweight",
    "High_Caloric_Food_Freq",
    "Calories_Monitoring",
    "SMOKE",
]


def load_all_processed_data():
    """
    Loads, cleans, splits, encodes, and scales the dataset in memory.
    This function contains ZERO blocking UI popup loops.
    """
    # 1. Load data
    df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

    # 2. Rename columns
    rename_dict = {
        "family_history_with_overweight": "Family_History_Overweight",
        "FAVC": "High_Caloric_Food_Freq",
        "FCVC": "Vegetable_Consumption_Freq",
        "NCP": "Main_Meals_Per_Day",
        "CAEC": "Food_Between_Meals",
        "CH2O": "Water_Intake_Daily",
        "SCC": "Calories_Monitoring",
        "FAF": "Physical_Activity_Freq",
        "TUE": "Tech_Device_Usage_Time",
        "CALC": "Alcohol_Consumption",
        "MTRANS": "Transportation_Method",
        "NObeyesdad": "Obesity_Level", 
    }
    df = df.rename(columns=rename_dict)

    # 3. Drop exact duplicates
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    # 4. Custom binary & ordinal mappings
    df["Gender"] = df["Gender"].map(GENDER_MAPPING)
    df["Alcohol_Consumption"] = df["Alcohol_Consumption"].map(HABIT_MAPPING)
    df["Food_Between_Meals"] = df["Food_Between_Meals"].map(HABIT_MAPPING)

    # 5. Isolate target
    X = df.drop(columns=["Obesity_Level"])
    y = df["Obesity_Level"].map(target_mapping)

    # 6. Train-Test Split (80/20 with stratification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7. One-Hot Encoding (fit categories on train only, then align test)
    X_train_encoded = pd.get_dummies(X_train, columns=CATEGORICAL_COLS, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, columns=CATEGORICAL_COLS, drop_first=True)

    X_train_encoded, X_test_encoded = X_train_encoded.align(
        X_test_encoded, join="left", axis=1, fill_value=0
    )

    # 8. Feature Scaling (fit on train numerical columns only)
    scaler = StandardScaler()
    X_train_scaled = X_train_encoded.copy()
    X_test_scaled = X_test_encoded.copy()

    X_train_scaled[NUMERICAL_COLS] = scaler.fit_transform(X_train_scaled[NUMERICAL_COLS])
    X_test_scaled[NUMERICAL_COLS] = scaler.transform(X_test_scaled[NUMERICAL_COLS])

    # Export your unscaled test tracking csv automatically right before returning variables
    test_export = X_test_encoded.copy()
    test_export["Obesity_Level"] = y_test.values
    test_export.to_csv("obesity_test_inputs_unscaled.csv", index=False)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X


def _to_yes_no(value):
    if pd.isna(value):
        return "no"
    if isinstance(value, str):
        return value.strip().lower()
    return "yes" if int(value) == 1 else "no"


def _to_habit(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    key = str(value).strip()
    if key in HABIT_MAPPING:
        return HABIT_MAPPING[key]
    return HABIT_MAPPING[key.lower()]


def _to_gender(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    key = str(value).strip()
    if key in GENDER_MAPPING:
        return GENDER_MAPPING[key]
    lowered = key.lower()
    if lowered == "male":
        return 1
    if lowered == "female":
        return 0
    return int(value)


def prepare_inference_features(single_input_df, expected_columns, scaler):
    """
    Encode one user row the same way as load_all_processed_data, then scale
    with the train-fitted scaler. Dummy columns are created without drop_first
    and then aligned to the training schema so a single row cannot drop the
    only present category.
    """
    df = pd.DataFrame(single_input_df).copy()

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map(_to_gender)

    if "Alcohol_Consumption" in df.columns:
        df["Alcohol_Consumption"] = df["Alcohol_Consumption"].map(_to_habit)
    if "Food_Between_Meals" in df.columns:
        df["Food_Between_Meals"] = df["Food_Between_Meals"].map(_to_habit)

    for col in YES_NO_COLS:
        if col in df.columns:
            df[col] = df[col].map(_to_yes_no)

    present_cats = [col for col in CATEGORICAL_COLS if col in df.columns]
    encoded = pd.get_dummies(df, columns=present_cats, drop_first=False)
    encoded = encoded.reindex(columns=list(expected_columns), fill_value=0)
    encoded = encoded.astype(float)

    scaled_cols = [col for col in NUMERICAL_COLS if col in encoded.columns]
    if scaled_cols:
        encoded[scaled_cols] = scaler.transform(encoded[scaled_cols])

    return encoded


def run_prediction(model, scaler, expected_columns, single_input_df):
    """Return (class index 0-6, probability vector aligned to CLASS_LABELS)."""
    processed_input = prepare_inference_features(single_input_df, expected_columns, scaler)
    predicted = int(model.predict(processed_input)[0])
    proba_row = model.predict_proba(processed_input)[0]
    ordered = [0.0] * len(CLASS_LABELS)
    for i, cls in enumerate(model.classes_):
        idx = int(cls)
        if 0 <= idx < len(ordered):
            ordered[idx] = float(proba_row[i])
    return predicted, ordered


def evaluate_classifier(model, X_test, y_test):
    """Accuracy (%), macro precision/recall/F1, OvR ROC-AUC, and a full 7x7 confusion matrix."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    accuracy = float(model.score(X_test, y_test) * 100)
    precision = float(precision_score(y_test, y_pred, average="macro", labels=CLASS_LABELS, zero_division=0))
    recall = float(recall_score(y_test, y_pred, average="macro", labels=CLASS_LABELS, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="macro", labels=CLASS_LABELS, zero_division=0))
    roc_auc = float(
        roc_auc_score(
            y_test,
            y_pred_proba,
            multi_class="ovr",
            average="macro",
            labels=CLASS_LABELS,
        )
    )
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS).tolist()
    return accuracy, precision, recall, f1, roc_auc, cm


def missing_values_heatmap_chart(df):
    """Missing-values heatmap on the RAW dataset (before any preprocessing)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", yticklabels=False, ax=ax)
    ax.set_title("Data Inspection: Missing Values Heatmap")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig


def duplicates_bar_chart(df):
    """Unique vs duplicate row counts on the RAW dataset (before de-duplication)."""
    duplicate_count = df.duplicated().sum()
    total_rows = len(df)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        x=["Unique Rows", "Duplicate Rows"],
        y=[total_rows - duplicate_count, duplicate_count],
        palette=["#4CAF50", "#FF5252"],
        ax=ax,
    )
    ax.set_title(f"Data Inspection: Duplicate Rows (Found {duplicate_count})")
    ax.set_ylabel("Number of Rows")
    fig.tight_layout()
    return fig


def scaling_before_after_chart(df):
    """
    KDE distributions of Weight/Height/Age before vs after StandardScaler.
    Runs its own local rename + encode + scale copy purely for visualization —
    this scaler is independent of the one used in load_all_processed_data().
    """
    rename_dict = {
        "family_history_with_overweight": "Family_History_Overweight",
        "FAVC": "High_Caloric_Food_Freq",
        "FCVC": "Vegetable_Consumption_Freq",
        "NCP": "Main_Meals_Per_Day",
        "CAEC": "Food_Between_Meals",
        "CH2O": "Water_Intake_Daily",
        "SCC": "Calories_Monitoring",
        "FAF": "Physical_Activity_Freq",
        "TUE": "Tech_Device_Usage_Time",
        "CALC": "Alcohol_Consumption",
        "MTRANS": "Transportation_Method",
        "NObeyesdad": "Obesity_Level",
    }
    df = df.rename(columns=rename_dict).drop_duplicates()

    integer_attributes = [
        "Vegetable_Consumption_Freq", "Main_Meals_Per_Day", "Water_Intake_Daily",
        "Physical_Activity_Freq", "Tech_Device_Usage_Time", "Age",
    ]
    for col in integer_attributes:
        df[col] = df[col].round().astype(int)
    df["Height"], df["Weight"] = df["Height"].round(2), df["Weight"].round(2)

    habit_mapping = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    df["Alcohol_Consumption"] = df["Alcohol_Consumption"].map(habit_mapping)
    df["Food_Between_Meals"] = df["Food_Between_Meals"].map(habit_mapping)
    df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})

    X_temp = pd.get_dummies(
        df.drop(columns=["Obesity_Level"]),
        columns=["Family_History_Overweight", "High_Caloric_Food_Freq", "Calories_Monitoring", "Transportation_Method", "SMOKE"],
        drop_first=True,
    )

    scaler = StandardScaler()
    numerical_cols = [
        "Age", "Height", "Weight", "Vegetable_Consumption_Freq", "Main_Meals_Per_Day",
        "Water_Intake_Daily", "Physical_Activity_Freq", "Tech_Device_Usage_Time",
        "Alcohol_Consumption", "Food_Between_Meals",
    ]

    X_scaled_temp = X_temp.copy()
    X_scaled_temp[numerical_cols] = scaler.fit_transform(X_scaled_temp[numerical_cols])

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
    ax1.set_title('Features BEFORE StandardScaler')
    sns.kdeplot(X_temp['Weight'], ax=ax1, color='r', label='Weight')
    sns.kdeplot(X_temp['Height'], ax=ax1, color='b', label='Height')
    sns.kdeplot(X_temp['Age'], ax=ax1, color='g', label='Age')
    ax1.set_xlabel('Original Units Range')
    ax1.legend()

    ax2.set_title('Features AFTER StandardScaler')
    sns.kdeplot(X_scaled_temp['Weight'], ax=ax2, color='r', label='Weight')
    sns.kdeplot(X_scaled_temp['Height'], ax=ax2, color='b', label='Height')
    sns.kdeplot(X_scaled_temp['Age'], ax=ax2, color='g', label='Age')
    ax2.set_xlabel('Standardized Z-Score Unit Scale (-3 to +3)')
    ax2.legend()
    fig.tight_layout()
    return fig


def correlation_heatmap_chart(df):
    """
    Correlation heatmap on the RAW dataset's already-numeric columns
    (Age, Height, Weight, FCVC, NCP, CH2O, FAF, TUE) — computed BEFORE any
    renaming, encoding, or scaling. Categorical raw columns (Gender, MTRANS,
    CALC, CAEC, SMOKE, family_history_with_overweight, FAVC, SCC) are excluded
    here since numeric_only=True drops them at this stage.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap (Raw Numeric Features)")
    fig.tight_layout()
    return fig


def visualize_dataset_analysis():
    """
    Run this function independently in an isolated script execution block
    to render your data profile plots on your desktop screen.
    """
    # Fresh temporary load to construct pristine data distributions
    df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

    missing_values_heatmap_chart(df)
    plt.show()

    duplicates_bar_chart(df)
    plt.show()

    scaling_before_after_chart(df)
    plt.show()