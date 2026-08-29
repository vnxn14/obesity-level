import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from ANN import get_trained_ann_model, predict_ann
from KNN import get_trained_knn_model, predict_knn
from DecisionTree import get_trained_decision_tree_model, predict_decision_tree
from SVM import get_trained_svm_model, predict_svm

# Page configurations
st.set_page_config(page_title="Obesity Risk Predictor", layout="wide")
st.title("🏋️‍♂️ Multi-Model Obesity Risk Prediction Dashboard")

# --- CUSTOM CSS FOR THE APP & CONFUSION MATRIX ---
st.markdown("""
<style>
.result-label {
    font-size: 13px;
    color: #6b6b6b;
    margin-bottom: 2px;
}
.result-value {
    font-size: 24px;
    font-weight: 600;
    margin: 0;
}
.bmi-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
}
.bmi-table th {
    text-align: left;
    color: #6b6b6b;
    font-weight: 500;
    font-size: 12px;
    padding: 6px 10px;
    border-bottom: 1px solid #e6e6e6;
}
.bmi-table td {
    padding: 8px 10px;
    border-bottom: 1px solid #f0f0f0;
}
.bmi-table tr.highlight td {
    background-color: #FAEEDA;
    font-weight: 600;
}

/* Confusion Matrix Dashboard Layout */
.metric-header-box {
    text-align: center;
    background-color: #f8f9fa;
    padding: 10px;
    border-radius: 8px;
    border: 1px solid #e9ecef;
}
.metric-header-title {
    font-size: 13px;
    color: #6c757d;
    margin-bottom: 2px;
}
.metric-header-val {
    font-size: 22px;
    font-weight: bold;
    color: #212529;
}
.cm-table {
    width: 100%;
    table-layout: fixed; /* 💡 CRITICAL: Forces all columns to obey identical widths */
    border-collapse: separate;
    border-spacing: 6px;
    font-size: 13px;
    text-align: center;
}
.cm-header {
    font-weight: 600;
    color: #495057;
    padding: 6px 2px;
    vertical-align: middle;
    word-wrap: break-word; /* 💡 Allows long words to wrap to a new line cleanly */
}
.cm-label-cell {
    font-weight: 600;
    color: #495057;
    text-align: right;
    padding-right: 12px;
    width: 130px; /* Gives the row labels on the far left their own stable width */
    vertical-align: middle;
}
.cm-cell-match, .cm-cell-miss, .cm-cell-zero {
    height: 65px; /* 💡 Forces a uniform square height for every cell */
    vertical-align: middle;
    border-radius: 6px;
    padding: 6px 2px;
}
.cm-cell-match {
    background-color: #2ecc71;
    color: white;
    font-weight: bold;
}
.cm-cell-match .pct {
    font-size: 10px;
    font-weight: normal;
    opacity: 0.85;
}
.cm-cell-miss {
    background-color: #fceade;
    color: #c0392b;
}
.cm-cell-miss .pct {
    font-size: 10px;
    color: #e67e22;
}
.cm-cell-zero {
    background-color: #f1f2f6;
    color: #a4b0be;
}
.cm-cell-zero .pct {
    font-size: 10px;
    opacity: 0.5;
}
</style>
""", unsafe_allow_html=True)

# --- GLOBAL VARIABLES & DATA CONFIGURATIONS ---
MATRIX_ORDER = [
    'Insufficient Weight', 'Normal Weight', 'Overweight Level I', 
    'Overweight Level II', 'Obesity Type I', 'Obesity Type II', 'Obesity Type III'
]

MODEL_CLASSES = [
    'Insufficient Weight',
    'Normal Weight',
    'Overweight Level I',
    'Overweight Level II',
    'Obesity Type I',
    'Obesity Type II',
    'Obesity Type III'
]

BMI_RANGES = {
    'Insufficient Weight': 'Below 18.5', 'Normal Weight': '18.5 – 24.9',
    'Overweight Level I': '25.0 – 27.4', 'Overweight Level II': '27.5 – 29.9',
    'Obesity Type I': '30.0 – 34.9', 'Obesity Type II': '35.0 – 39.9', 'Obesity Type III': '40.0 and above',
}

GAUGE_COLORS = [
    "#85B7EB",  # Insufficient Weight (Blue)
    "#2ecc71",  # Normal Weight (Green)
    "#f1c40f",  # Overweight Level I (Yellow)
    "#e67e22",  # Overweight Level II (Orange)
    "#e74c3c",  # Obesity Type I (Light Red)
    "#c0392b",  # Obesity Type II (Medium Red)
    "#962d22"   # Obesity Type III (Dark Red)
]
GAUGE_STOPS = [
    0.0,
    16.0,
    33.0,
    42.0,
    54.0,
    68.0,
    83.0,
    100.0
]
# Metrics database mapping metrics cleanly to each model configuration block
MODEL_PERFORMANCE_METRICS = {
    "ANN (Artificial Neural Network)": {"Accuracy": "89.2%", "F1-Score": "0.89", "ROC-AUC": "0.94"},
    "KNN (K-Nearest Neighbors)": {"Accuracy": "82.4%", "F1-Score": "0.82", "ROC-AUC": "0.88"},
    "SVM (Support Vector Machine)": {"Accuracy": "86.7%", "F1-Score": "0.86", "ROC-AUC": "0.92"},
    "Decision Tree": {"Accuracy": "84.1%", "F1-Score": "0.84", "ROC-AUC": "0.89"}
}

# Dummy confusion matrix subsets simplified to a clean 4x4 matching your visual requirements
MATRIX_LABELS = [
    'Insufficient Weight',
    'Normal Weight',
    'Overweight Level I',
    'Overweight Level II',
    'Obesity Type I',
    'Obesity Type II',
    'Obesity Type III'
]

# --- GAUGE GENERATION HELPER ---
def get_band_color(position_pct: float) -> str:
    """Returns the gauge color band a given position (0-100) falls into."""
    for i in range(len(GAUGE_STOPS) - 1):
        if GAUGE_STOPS[i] <= position_pct <= GAUGE_STOPS[i + 1]:
            return GAUGE_COLORS[min(i, len(GAUGE_COLORS)-1)]
    return GAUGE_COLORS[-1]

def reorder_probabilities(model, probabilities):
    probabilities = np.asarray(probabilities)

    if probabilities.ndim > 1:
        probabilities = probabilities[0]

    ordered_probs = np.zeros(len(MODEL_CLASSES))

    for model_index, class_name in enumerate(model.classes_):
        if class_name in MODEL_CLASSES:
            target_index = MODEL_CLASSES.index(class_name)
            ordered_probs[target_index] = probabilities[model_index]

    return ordered_probs

def calculate_gauge_position(probs):
    probs = np.asarray(probs, dtype=float)

    # Normalize probabilities
    total = probs.sum()

    if total <= 0:
        return 50.0

    probs = probs / total

    # Gauge category centers
    centers = np.array([
        (GAUGE_STOPS[i] + GAUGE_STOPS[i + 1]) / 2
        for i in range(len(GAUGE_STOPS) - 1)
    ])

    # Weighted average of category positions
    position = np.dot(probs, centers)

    return float(np.clip(position, 0, 100))

def render_gauge(position_pct: float, label: str):
    fig = go.Figure()
    for i in range(len(GAUGE_STOPS) - 1):
        fig.add_shape(
            type="rect", x0=GAUGE_STOPS[i], x1=GAUGE_STOPS[i + 1], y0=0, y1=1,
            fillcolor=GAUGE_COLORS[min(i, len(GAUGE_COLORS)-1)], line=dict(width=0),
        )
    fig.add_shape(type="line", x0=position_pct, x1=position_pct, y0=-0.25, y1=1.25, line=dict(color="black", width=3))
    xanchor = "right" if position_pct >= 85 else ("left" if position_pct <= 15 else "center")
    fig.add_annotation(x=position_pct, y=1.5, showarrow=False, text=f"<b>{label}</b>", font=dict(size=13), xanchor=xanchor)
    fig.update_xaxes(range=[-8, 108], visible=False)
    fig.update_yaxes(range=[-0.5, 2], visible=False)
    fig.update_layout(height=100, margin=dict(l=10, r=10, t=25, b=10), plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
    return fig

def render_probability_bar(probs: np.ndarray, predicted_idx: int, highlight_color: str):
    colors = [highlight_color if i == predicted_idx else "#d9d9d9" for i in range(len(MATRIX_LABELS))]
    fig = go.Figure(go.Bar(x=probs * 100, y=MATRIX_LABELS, orientation="h", marker_color=colors, text=[f"{p*100:.1f}%" for p in probs], textposition="outside"))
    fig.update_layout(height=260, margin=dict(l=10, r=40, t=10, b=10), xaxis=dict(title="Probability (%)", range=[0, 100]), yaxis=dict(autorange="reversed"), plot_bgcolor="white", paper_bgcolor="white")
    return fig

# --- DYNAMIC HTML CONFUSION MATRIX INLINE COMPONENT ---
def generate_html_confusion_matrix(matrix_array):
    row_sums = [sum(r) for r in matrix_array]
    
    table_html = '<table class="cm-table">'
    table_html += '<tr><th colspan="8" class="cm-header" style="text-align: center; font-size: 15px; padding-bottom: 10px;">Predicted Classification</th></tr>'
    table_html += '<tr><th class="cm-header" style="text-align:right; padding-right:15px;">True \\ Pred</th>'
    for label in MATRIX_LABELS:
        table_html += f'<th class="cm-header">{label}</th>'
    table_html += '</tr>'
    
    for i, row_label in enumerate(MATRIX_LABELS):
        table_html += f'<tr><td class="cm-label-cell">{row_label}</td>'
        for j, val in enumerate(matrix_array[i]):
            total_in_row = row_sums[i] if row_sums[i] > 0 else 1
            pct = int((val / total_in_row) * 100)
            
            if i == j:
                cell_class = "cm-cell-match"
            elif val > 0:
                cell_class = "cm-cell-miss"
            else:
                cell_class = "cm-cell-zero"
                
            table_html += f'<td class="{cell_class}"><div>{val}</div><div class="pct">{pct}%</div></td>'
        table_html += '</tr>'
        
    table_html += '</table>'
    return table_html

# --- SCREEN SPLIT LAYOUT (LEFT: OUTPUTS / VIEWS, RIGHT: INPUT PANEL) ---
main_output_col, main_input_col = st.columns([1.2, 1.0], gap="large")
# ==========================================
# RIGHT COLUMN: ALL USER SETTINGS & FORM PARAMETERS
# ==========================================
with main_input_col:
    st.subheader("👤 Model Selection & Patient Attributes")
    
    selected_model = st.selectbox(
        "Choose an Active Machine Learning Model",
        ["ANN (Artificial Neural Network)", "KNN (K-Nearest Neighbors)", "SVM (Support Vector Machine)", "Decision Tree"],
        index=0
    )
    
    st.markdown("---")
    
    in_col1, in_col2 = st.columns(2)
    with in_col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Age", min_value=1, max_value=120, value=25)
        height = st.number_input("Height (meters)", min_value=1.0, max_value=2.5, value=1.76, step=0.01)
        weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=74.0, step=0.1)
        family_history = st.selectbox("Family History with Overweight?", ["Yes", "No"])
        high_caloric = st.selectbox("Frequently Consume High Caloric Food?", ["Yes", "No"])
        veg_cons = st.slider("Vegetables Consumption Frequency (0-3)", 0.0, 3.0, 2.0, step=0.01)
        main_meals = st.slider("Number of Main Meals Daily (1-4)", 1.0, 4.0, 3.0, step=0.01)

    with in_col2:
        food_between = st.selectbox("Eat Food Between Meals?", ["No", "Sometimes", "Frequently", "Always"])
        smoke_status = st.selectbox("Do you smoke?", options=["No", "Yes"])
        water = st.slider("Daily Water Intake (Liters, 1-3)", 1.0, 3.0, 2.0, step=0.01)
        calorie_mon = st.selectbox("Monitor Calories Consumed Daily?", ["Yes", "No"])
        phys_act = st.slider("Physical Activity Frequency (0-3)", 0.0, 3.0, 1.0, step=0.01)
        tech_usage = st.slider("Time Using Tech Devices (0-2 hours)", 0.0, 2.0, 1.0, step=0.01)
        alcohol = st.selectbox("Alcohol Intake Frequency?", ["No", "Sometimes", "Frequently", "Always"])
        transport = st.selectbox("Primary Transportation Mode", ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"])

    # Feature engineering compilation
    raw_input_data = {
        'Gender': 1 if gender.lower() == 'male' else 0,
        'Age': int(np.round(age, 0)), 'Height': round(height, 2), 'Weight': round(weight, 1),
        'Family_History': 1 if family_history.lower() == 'yes' else 0,
        'High_Caloric_Food': 1 if high_caloric.lower() == 'yes' else 0,
        'Veg_Consumption': round(veg_cons, 2), 
        'Num_Main_Meals': round(main_meals, 2),
        'Food_Between_Meals': {'no': 0, 'sometimes': 1, 'frequently': 2, 'always': 3}[food_between.lower()],
        'SMOKE': 1 if smoke_status.lower() == 'yes' else 0, 
        'Water_Intake': round(water, 2),
        'Calorie_Monitoring': 1 if calorie_mon.lower() == 'yes' else 0, 
        'Physical_Activity': round(phys_act, 2),
        'Tech_Usage_Time': round(tech_usage, 2), 
        'Alcohol_Intake': {'no': 0, 'sometimes': 1, 'frequently': 2, 'always': 3}[alcohol.lower()]}
    calculated_bmi = round(weight / (height ** 2), 2)
    for mode in ['Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking']:
        raw_input_data[f"MTRANS_{mode}"] = 1 if transport.lower() == mode.lower() else 0
    input_df = pd.DataFrame([raw_input_data])
# ==========================================
# LEFT COLUMN: THE TOGGLE TABS (PREDICTION VS MATRIX)
# ==========================================
with main_output_col:
    # 🎯 Creates the exact side-by-side tabs you wanted from your mockup images
    view_tab1, view_tab2 = st.tabs(["🔵 Prediction Simulator", "📊 Confusion Matrix Analytics"])

    # ------------------ TAB 1: PREDICTION SIMULATOR ------------------
    with view_tab1:
        st.subheader("📊 Live Prediction Classification Results")
        # Load active model pipeline outputs dynamically using stateful caching
        prediction_index, probabilities = 0, None
        accuracy_score, f1_score_val, roc_auc_val = 0.0, 0.0, 0.0
        actual_cm_array = [[0] * len(MATRIX_LABELS) for _ in MATRIX_LABELS]  # safe all-zero default

        try:
            if selected_model == "ANN (Artificial Neural Network)":
                with st.spinner("Calculating via ANN..."):
                    model, scaler, cols, acc, f1, auc, cm = get_trained_ann_model()
                    prediction_index, probabilities = predict_ann(model, scaler, cols, input_df)
                    accuracy_score, f1_score_val, roc_auc_val, actual_cm_array = acc, f1, auc, cm

            elif selected_model == "KNN (K-Nearest Neighbors)":
                with st.spinner("Calculating via KNN..."):
                    model, scaler, cols, acc, f1, auc, cm = get_trained_knn_model()
                    prediction_index, probabilities = predict_knn(model, scaler, cols, input_df)
                    accuracy_score, f1_score_val, roc_auc_val, actual_cm_array = acc, f1, auc, cm

            elif selected_model == "SVM (Support Vector Machine)":
                with st.spinner("Calculating via SVM..."):
                    model, scaler, cols, acc, f1, auc, cm = get_trained_svm_model()
                    prediction_index, probabilities = predict_svm(model, scaler, cols, input_df)
                    accuracy_score, f1_score_val, roc_auc_val, actual_cm_array = acc, f1, auc, cm

            elif selected_model == "Decision Tree":
                with st.spinner("Calculating via Decision Tree..."):
                    model, scaler, cols, acc, f1, auc, cm = get_trained_decision_tree_model()
                    prediction_index, probabilities = predict_decision_tree(model, scaler, cols, input_df)
                    accuracy_score, f1_score_val, roc_auc_val, actual_cm_array = acc, f1, auc, cm
        except Exception as e:
            # Surface real failures instead of silently substituting fabricated
            # numbers — a hidden bug reported as a confident fake result is worse
            # than a visible error.
            st.error(f"⚠️ Prediction failed for {selected_model}: {e}")
            st.stop()

        if hasattr(probabilities, "ndim") and probabilities.ndim > 1:
            probabilities = probabilities[0]

        probs_array = np.asarray(probabilities)
        corrected_prediction_idx = int(np.argmax(probs_array))
        predicted_label = MATRIX_LABELS[corrected_prediction_idx]

        position_pct = calculate_gauge_position(probs_array)

        band_color = GAUGE_COLORS[corrected_prediction_idx]

        prediction_confidence = (
            probs_array[corrected_prediction_idx] * 100
        )
        with st.container(border=True):
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.markdown('<p class="result-label">Predicted Category</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="result-value">{predicted_label}</p>', unsafe_allow_html=True)
            with m_col2:
                st.markdown('<p class="result-label">Model Accuracy</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="result-value">{accuracy_score:.1f}%</p>', unsafe_allow_html=True)
            with m_col3:
                st.markdown('<p class="result-label">Live BMI Score</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="result-value">{calculated_bmi:.2f}</p>', unsafe_allow_html=True)
            st.plotly_chart(render_gauge(position_pct, predicted_label), use_container_width=True)
            st.markdown('<p class="result-label">Probability Distribution Profile</p>', unsafe_allow_html=True)
            st.plotly_chart(render_probability_bar(probs_array, corrected_prediction_idx, band_color), use_container_width=True)
    # ------------------ TAB 2: CONFUSION MATRIX VIEWER ------------------
    with view_tab2:
        st.subheader("🎯 Evaluation Performance Metrics")

        # Header evaluation score cards matching your second and third mockup layout
        h_col1, h_col2, h_col3 = st.columns(3)
        with h_col1:
            st.markdown(f'<div class="metric-header-box"><div class="metric-header-title">Accuracy</div><div class="metric-header-val">{accuracy_score:.1f}%</div></div>', unsafe_allow_html=True)
        with h_col2:
            st.markdown(f'<div class="metric-header-box"><div class="metric-header-title">F1-Score</div><div class="metric-header-val">{f1_score_val:.2f}</div></div>', unsafe_allow_html=True)
        with h_col3:
            st.markdown(f'<div class="metric-header-box"><div class="metric-header-title">ROC-AUC</div><div class="metric-header-val">{roc_auc_val:.2f}</div></div>', unsafe_allow_html=True)
            
        st.markdown(generate_html_confusion_matrix(actual_cm_array), unsafe_allow_html=True)