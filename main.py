import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse, Polygon

from ANN import get_trained_ann_model, predict_ann
from KNN import get_trained_knn_model, predict_knn
from DecisionTree import get_trained_decision_tree_model, predict_decision_tree
from SVM import get_trained_svm_model, predict_svm

# Page configurations
st.set_page_config(page_title="Obesity Risk Predictor", page_icon="🏋️", layout="wide")
st.title("🏋️‍♂️ Multi-Model Obesity Risk Prediction Dashboard")
st.caption("Adjust your profile on the right — every chart on the left updates live as you type.")

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
    table-layout: fixed;
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
    word-wrap: break-word;
}
.cm-label-cell {
    font-weight: 600;
    color: #495057;
    text-align: right;
    padding-right: 12px;
    width: 130px;
    vertical-align: middle;
}
.cm-cell-match, .cm-cell-miss, .cm-cell-zero {
    height: 65px;
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
MATRIX_LABELS = [
    'Insufficient Weight', 'Normal Weight', 'Overweight Level I',
    'Overweight Level II', 'Obesity Type I', 'Obesity Type II', 'Obesity Type III'
]

GAUGE_COLORS = ["#85B7EB", "#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#c0392b", "#962d22"]

GAUGE_STOPS = list(np.linspace(0, 100, len(GAUGE_COLORS) + 1))

TRANSPORT_MODES = [
    {"name": "Automobile", "icon": "🚗", "desc": "Private car commute", "code": "Automobile"},
    {"name": "Bike", "icon": "🚲", "desc": "Active cycling", "code": "Bike"},
    {"name": "Motorbike", "icon": "🛵", "desc": "Two-wheel motor ride", "code": "Motorbike"},
    {"name": "Transit", "icon": "🚌", "desc": "Bus, metro, or rail", "code": "Public_Transportation"},
    {"name": "Walking", "icon": "🚶", "desc": "Pedestrian commute", "code": "Walking"},
]

if "transport_mode" not in st.session_state:
    st.session_state.transport_mode = "Automobile"

# --- GAUGE / VISUAL HELPERS ---
def calculate_gauge_position(predicted_idx: int) -> float:
    idx = int(np.clip(predicted_idx, 0, len(GAUGE_COLORS) - 1))
    return (GAUGE_STOPS[idx] + GAUGE_STOPS[idx + 1]) / 2


def render_gauge(position_pct: float, label: str):
    fig = go.Figure()
    for i in range(len(GAUGE_STOPS) - 1):
        fig.add_shape(
            type="rect", x0=GAUGE_STOPS[i], x1=GAUGE_STOPS[i + 1], y0=0, y1=1,
            fillcolor=GAUGE_COLORS[i], line=dict(width=0),
        )
    fig.add_shape(type="line", x0=position_pct, x1=position_pct, y0=-0.25, y1=1.25, line=dict(color="black", width=3))
    xanchor = "right" if position_pct >= 85 else ("left" if position_pct <= 15 else "center")
    fig.add_annotation(x=position_pct, y=1.5, showarrow=False, text=f"<b>{label}</b>", font=dict(size=13), xanchor=xanchor)
    fig.update_xaxes(range=[-5, 105], visible=False)
    fig.update_yaxes(range=[-0.5, 2], visible=False)
    fig.update_layout(height=80, margin=dict(l=10, r=10, t=22, b=8), plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
    return fig


def render_probability_bar(probs: np.ndarray, predicted_idx: int, highlight_color: str):
    colors = [highlight_color if i == predicted_idx else "#d9d9d9" for i in range(len(MATRIX_LABELS))]
    fig = go.Figure(go.Bar(x=probs * 100, y=MATRIX_LABELS, orientation="h", marker_color=colors, text=[f"{p*100:.1f}%" for p in probs], textposition="outside"))
    fig.update_layout(height=200, margin=dict(l=10, r=40, t=10, b=10), xaxis=dict(title="Probability (%)", range=[0, 100]), yaxis=dict(autorange="reversed"), plot_bgcolor="white", paper_bgcolor="white")
    return fig


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
            cell_class = "cm-cell-match" if i == j else ("cm-cell-miss" if val > 0 else "cm-cell-zero")
            table_html += f'<td class="{cell_class}"><div>{val}</div><div class="pct">{pct}%</div></td>'
        table_html += '</tr>'
    table_html += '</table>'
    return table_html


def get_bmi_category(bmi: float):
    if bmi < 18.5:
        return "Underweight", "#f1c40f"
    elif bmi < 25.0:
        return "Normal Weight", "#2ecc71"
    elif bmi < 30.0:
        return "Overweight", "#e67e22"
    return "Obese", "#e74c3c"


def render_bmi_dial(bmi: float, color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bmi,
        number={'suffix': " BMI", 'font': {'size': 26}},
        gauge={
            'axis': {'range': [13.0, 43.0], 'tickwidth': 1, 'tickcolor': "gray"},
            'bar': {'color': color, 'thickness': 0.3},
            'steps': [
                {'range': [13.0, 18.5], 'color': 'rgba(241, 196, 15, 0.35)'},
                {'range': [18.5, 25.0], 'color': 'rgba(46, 204, 113, 0.35)'},
                {'range': [25.0, 30.0], 'color': 'rgba(230, 126, 34, 0.35)'},
                {'range': [30.0, 43.0], 'color': 'rgba(231, 76, 60, 0.35)'},
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': bmi},
        },
    ))
    fig.update_layout(height=180, margin=dict(l=20, r=20, t=25, b=5), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def render_avatar(height_m: float, weight_kg: float, color: str):
    scale_h = height_m / 1.65

    head_cy = 205.0
    head_r = 14.0
    neck_w, neck_h = 7.0, 7.0
    torso_top = head_cy - head_r - neck_h
    torso_h = 58.0 * scale_h
    torso_bottom = torso_top - torso_h
    leg_h = 80.0 * scale_h
    leg_bottom = torso_bottom - leg_h

    skin_color = '#f2c7a5'
    pants_color = '#2c3e50'
    hair_color = '#2c3e50'

    body_base = max(18.0, min(28.0 + (weight_kg - 55.0) * 0.38, 62.0))
    shoulder_w = body_base * 1.22
    waist_w = body_base * 0.95
    hip_w = body_base * 1.0

    arm_thick = max(5.5, min(13.0, 7.0 + (weight_kg - 55.0) * 0.08))
    arm_offset = shoulder_w / 2.0 + arm_thick / 2.0 + 1.0
    leg_thick = max(7.0, min(16.0, 8.5 + (weight_kg - 55.0) * 0.10))
    leg_gap = max(2.5, hip_w * 0.16)

    fig, ax = plt.subplots(figsize=(0.96, 1.26), dpi=100)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    ax.add_patch(Ellipse((85, leg_bottom + 1), hip_w * 1.5, 7, color='#111111', alpha=0.35))
    ax.add_patch(Circle((85, head_cy), head_r, color=skin_color, zorder=3))
    ax.add_patch(Rectangle((85 - neck_w / 2, torso_top), neck_w, neck_h + 2, color=skin_color, zorder=2))
    ax.add_patch(Ellipse((85, head_cy + 5), head_r * 2.1, head_r * 1.2, color=hair_color, zorder=4))

    torso_coords = [
        (85 - shoulder_w / 2, torso_top),
        (85 + shoulder_w / 2, torso_top),
        (85 + waist_w / 2, torso_top - torso_h * 0.6),
        (85 + hip_w / 2, torso_bottom),
        (85 - hip_w / 2, torso_bottom),
        (85 - waist_w / 2, torso_top - torso_h * 0.6),
    ]
    ax.add_patch(Polygon(torso_coords, color=color, zorder=3))

    arm_h = 48.0 * scale_h
    ax.add_patch(Rectangle((85 - arm_offset - arm_thick / 2, torso_top - arm_h), arm_thick, arm_h, color=skin_color, zorder=2))
    ax.add_patch(Rectangle((85 + arm_offset - arm_thick / 2, torso_top - arm_h), arm_thick, arm_h, color=skin_color, zorder=2))
    ax.add_patch(Rectangle((85 - arm_offset - arm_thick / 2, torso_top - 14), arm_thick, 14, color=color, zorder=3))
    ax.add_patch(Rectangle((85 + arm_offset - arm_thick / 2, torso_top - 14), arm_thick, 14, color=color, zorder=3))

    ax.add_patch(Rectangle((85 - leg_gap - leg_thick, leg_bottom), leg_thick, leg_h, color=pants_color, zorder=2))
    ax.add_patch(Rectangle((85 + leg_gap, leg_bottom), leg_thick, leg_h, color=pants_color, zorder=2))
    ax.add_patch(Ellipse((85 - leg_gap - leg_thick / 2, leg_bottom), leg_thick * 1.3, 5, color='#2c3e50', zorder=3))
    ax.add_patch(Ellipse((85 + leg_gap + leg_thick / 2, leg_bottom), leg_thick * 1.3, 5, color='#2c3e50', zorder=3))

    ax.set_xlim(0, 170)
    ax.set_ylim(leg_bottom - 5, 230)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def metric_card(title: str, value: str):
    st.markdown(
        f'<div class="metric-header-box"><div class="metric-header-title">{title}</div>'
        f'<div class="metric-header-val">{value}</div></div>',
        unsafe_allow_html=True,
    )


# --- SCREEN SPLIT LAYOUT ---
main_output_col, main_input_col = st.columns([1.2, 1.0], gap="large")

with main_input_col:
    st.subheader("👤 Model Selection & Patient Attributes")
    selected_model = st.selectbox(
        "Choose an Active Machine Learning Model",
        ["ANN (Artificial Neural Network)", "KNN (K-Nearest Neighbors)", "SVM (Support Vector Machine)", "Decision Tree"],
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
        veg_cons = st.slider("Vegetables Consumption Frequency", 0.0, 3.0, 2.0)
        st.caption("0: Never  |  1: Sometimes  |  2: Often  |  3: Always")
        main_meals = st.slider("Number of Main Meals Daily", 1.0, 4.0, 3.0)
        st.caption("1: 1-2 meals  |  2: 3 meals  |  3-4: More than 3 meals")

    with in_col2:
        food_between = st.selectbox("Eat Food Between Meals?", ["No", "Sometimes", "Frequently", "Always"])
        smoke_status = st.selectbox("Do you smoke?", options=["No", "Yes"])
        water = st.slider("Daily Water Intake (Liters)", 1.0, 3.0, 2.0)
        calorie_mon = st.selectbox("Monitor Calories Consumed Daily?", ["Yes", "No"])
        phys_act = st.slider("Physical Activity Frequency", 0.0, 3.0, 1.0)
        st.caption("0: None  |  1: 1-2 days  |  2: 2-4 days  |  3: 4-5 days")
        tech_usage = st.slider("Time Using Tech Devices", 0.0, 2.0, 1.0)
        st.caption("0: 0-2 hours  |  1: 3-5 hours  |  2: > 5 hours")
        alcohol = st.selectbox("Alcohol Intake Frequency?", ["No", "Sometimes", "Frequently", "Always"])

    # Interactive pill-style transportation picker (replaces plain dropdown)
    st.write("**Primary Transportation Mode**")
    t_cols = st.columns(5)
    for idx, item in enumerate(TRANSPORT_MODES):
        with t_cols[idx]:
            is_active = st.session_state.transport_mode == item["code"]
            if st.button(f"{item['icon']} {item['name']}", key=f"t_btn_{item['code']}",
                         type="primary" if is_active else "secondary", use_container_width=True):
                st.session_state.transport_mode = item["code"]
                st.rerun()
            st.caption(item["desc"])
    transport = st.session_state.transport_mode

    raw_input_data = {
        "Gender": gender,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "Family_History_Overweight": family_history.lower(),
        "High_Caloric_Food_Freq": high_caloric.lower(),
        "Vegetable_Consumption_Freq": veg_cons,
        "Main_Meals_Per_Day": main_meals,
        "Food_Between_Meals": food_between,
        "SMOKE": smoke_status.lower(),
        "Water_Intake_Daily": water,
        "Calories_Monitoring": calorie_mon.lower(),
        "Physical_Activity_Freq": phys_act,
        "Tech_Device_Usage_Time": tech_usage,
        "Alcohol_Consumption": alcohol,
        "Transportation_Method": transport.replace("_", " "),
    }
    calculated_bmi = round(weight / (height ** 2), 2)
    input_df = pd.DataFrame([raw_input_data])

with main_output_col:
    view_tab1, view_tab3 = st.tabs(
        ["🔵 Live Assessment", "📊 Confusion Matrix Analytics"]
    )

    prediction_index, probabilities = 0, None
    accuracy_score, f1_score_val, roc_auc_val = 0.0, 0.0, 0.0
    actual_cm_array = [[0] * len(MATRIX_LABELS) for _ in MATRIX_LABELS]
    prediction_failed = False

    try:
        if selected_model == "ANN (Artificial Neural Network)":
            model, scaler, cols, acc, f1, auc, cm = get_trained_ann_model()
            prediction_index, probabilities = predict_ann(model, scaler, cols, input_df)
        elif selected_model == "KNN (K-Nearest Neighbors)":
            model, scaler, cols, acc, f1, auc, cm = get_trained_knn_model()
            prediction_index, probabilities = predict_knn(model, scaler, cols, input_df)
        elif selected_model == "SVM (Support Vector Machine)":
            model, scaler, cols, acc, f1, auc, cm = get_trained_svm_model()
            prediction_index, probabilities = predict_svm(model, scaler, cols, input_df)
        elif selected_model == "Decision Tree":
            model, scaler, cols, acc, f1, auc, cm = get_trained_decision_tree_model()
            prediction_index, probabilities = predict_decision_tree(model, scaler, cols, input_df)

        accuracy_score, f1_score_val, roc_auc_val, actual_cm_array = acc, f1, auc, cm
    except Exception as e:
        prediction_failed = True
        with view_tab1:
            st.error(f"⚠️ Prediction failed for {selected_model}: {e}")

    if not prediction_failed:
        if hasattr(probabilities, "ndim") and probabilities.ndim > 1:
            probabilities = probabilities[0]

        probs_array = np.asarray(probabilities, dtype=float)
        predicted_idx = int(prediction_index)
        predicted_label = MATRIX_LABELS[predicted_idx]
        position_pct = calculate_gauge_position(predicted_idx)
        band_color = GAUGE_COLORS[predicted_idx]
        bmi_category, bmi_color = get_bmi_category(calculated_bmi)

        with view_tab1:
            # --- Section 1: BMI dial + body silhouette, side by side ---
            st.subheader("📈 Live BMI & Body Snapshot")
            dial_col, avatar_col = st.columns([1.3, 1.0], gap="small", vertical_alignment="top")
            with dial_col:
                st.plotly_chart(render_bmi_dial(calculated_bmi, bmi_color), use_container_width=True)
            with avatar_col:
                st.pyplot(render_avatar(height, weight, bmi_color), use_container_width=False)

            # --- Section 2: model prediction gauge + probability bar ---
            st.subheader("📊 Live Prediction Classification Results")
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                metric_card("Predicted Category", predicted_label)
            with m_col2:
                metric_card("Model Accuracy", f"{accuracy_score:.1f}%")
            st.plotly_chart(render_gauge(position_pct, predicted_label), use_container_width=True)
            st.markdown('<div class="result-label">Probability Distribution Profile</div>', unsafe_allow_html=True)
            st.plotly_chart(render_probability_bar(probs_array, predicted_idx, band_color), use_container_width=True)

        with view_tab3:
            st.subheader("🎯 Evaluation Performance Metrics")
            h_col1, h_col2, h_col3 = st.columns(3)
            with h_col1:
                metric_card("Accuracy", f"{accuracy_score:.1f}%")
            with h_col2:
                metric_card("F1-Score", f"{f1_score_val:.2f}")
            with h_col3:
                metric_card("ROC-AUC", f"{roc_auc_val:.2f}")
            st.write("")
            st.markdown(generate_html_confusion_matrix(actual_cm_array), unsafe_allow_html=True)