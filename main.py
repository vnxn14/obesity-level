import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from ANN import predict_ann
from KNN import train_knn_model, predict_knn

# Page configurations
st.set_page_config(page_title="Obesity Risk Predictor", layout="wide")
st.title("🏋️‍♂️ Multi-Model Obesity Risk Prediction Dashboard")

# --- CUSTOM CSS ---
st.markdown("""
<style>
.result-label {
    font-size: 13px;
    color: #6b6b6b;
    margin-bottom: 2px;
}
.result-value {
    font-size: 26px;
    font-weight: 600;
    margin: 0;
}
.small-note {
    font-size: 12px;
    color: #9a9a9a;
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
</style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR CONFIGURATION ---
st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose a Machine Learning Model",
    ["ANN (Artificial Neural Network)", "KNN (K-Nearest Neighbors)", "SVM (Support Vector Machine)", "Decision Tree"],
    index=0  # Defaults selection directly to ANN
)

st.sidebar.markdown("---")
st.sidebar.info(f"Active Model: **{selected_model}**")

# --- 3. MAIN USER INPUT FORM ---
st.header("👤 Enter Individual Attributes & Habits")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    height = st.number_input("Height (meters)", min_value=1.0, max_value=2.5, value=1.75, step=0.01)
    weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.1)

with col2:
    family_history = st.selectbox("Family History with Overweight?", ["Yes", "No"])
    high_caloric = st.selectbox("Frequently Consume High Caloric Food?", ["Yes", "No"])
    veg_cons = st.slider("Vegetables Consumption Frequency (0-3)", 0.0, 3.0, 2.0, step=0.01)
    main_meals = st.slider("Number of Main Meals Daily (1-4)", 1.0, 4.0, 3.0, step=0.01)
    food_between = st.selectbox("Eat Food Between Meals?", ["No", "Sometimes", "Frequently", "Always"])

with col3:
    water = st.slider("Daily Water Intake (Liters, 1-3)", 1.0, 3.0, 2.0, step=0.01)
    calorie_mon = st.selectbox("Monitor Calories Consumed Daily?", ["Yes", "No"])
    phys_act = st.slider("Physical Activity Frequency (0-3)", 0.0, 3.0, 1.0, step=0.01)
    tech_usage = st.slider("Time Using Tech Devices (0-2 hours)", 0.0, 2.0, 1.0, step=0.01)
    alcohol = st.selectbox("Alcohol Intake Frequency?", ["No", "Sometimes", "Frequently", "Always"])
    transport = st.selectbox("Primary Transportation Mode", ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"])

# --- 4. INPUT DATA PREPROCESSING PIPELINE ---
raw_input_data = {
    'Gender': 1 if gender.lower() == 'male' else 0,
    'Age': int(np.round(age, 0)),
    'Height': round(height, 2),
    'Weight': round(weight, 1),
    'Family_History': 1 if family_history.lower() == 'yes' else 0,
    'High_Caloric_Food': 1 if high_caloric.lower() == 'yes' else 0,
    'Veg_Consumption': round(veg_cons, 2),
    'Num_Main_Meals': round(main_meals, 2),
    'Food_Between_Meals': {'no': 0, 'sometimes': 1, 'frequently': 2, 'always': 3}[food_between.lower()],
    'SMOKE': 0,
    'Water_Intake': round(water, 2),
    'Calorie_Monitoring': 1 if calorie_mon.lower() == 'yes' else 0,
    'Physical_Activity': round(phys_act, 2),
    'Tech_Usage_Time': round(tech_usage, 2),
    'Alcohol_Intake': {'no': 0, 'sometimes': 1, 'frequently': 2, 'always': 3}[alcohol.lower()],
    'BMI': round(weight / (height ** 2), 2)
}

# Construct matching One-Hot encoded structure for Transport Mode
transport_modes = ['Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking']
for mode in transport_modes:
    feature_name = f"MTRANS_{mode}"
    raw_input_data[feature_name] = 1 if transport.lower() == mode.lower() else 0

input_df = pd.DataFrame([raw_input_data])

CLASS_ORDER = [
    'Insufficient Weight', 'Normal Weight', 'Overweight Level I',
    'Overweight Level II', 'Obesity Type I', 'Obesity Type II', 'Obesity Type III'
]
target_reverse_map = {i: label for i, label in enumerate(CLASS_ORDER)}

# BMI reference ranges matching the dataset's class boundaries
BMI_RANGES = {
    'Insufficient Weight': 'Below 18.5',
    'Normal Weight': '18.5 – 24.9',
    'Overweight Level I': '25.0 – 27.4',
    'Overweight Level II': '27.5 – 29.9',
    'Obesity Type I': '30.0 – 34.9',
    'Obesity Type II': '35.0 – 39.9',
    'Obesity Type III': '40.0 and above',
}

# Color spectrum matched to class order (blue -> green -> amber -> coral -> red)
GAUGE_COLORS = ["#85B7EB", "#639922", "#BA7517", "#D85A30", "#A32D2D"]
GAUGE_STOPS = [0, 20, 45, 70, 100]  # boundaries as % of the 0-100 axis


def render_gauge(position_pct: float, label: str):
    """Horizontal spectrum gauge with a marker at the predicted position."""
    fig = go.Figure()

    for i in range(len(GAUGE_STOPS) - 1):
        fig.add_shape(
            type="rect",
            x0=GAUGE_STOPS[i], x1=GAUGE_STOPS[i + 1],
            y0=0, y1=1,
            fillcolor=GAUGE_COLORS[i],
            line=dict(width=0),
        )

    fig.add_shape(
        type="line",
        x0=position_pct, x1=position_pct,
        y0=-0.25, y1=1.25,
        line=dict(color="black", width=3),
    )

    # Anchor the label away from the marker near either edge so long labels
    # (e.g. "Insufficient Weight", "Obesity Type III") don't get clipped by
    # the plot boundary when the marker sits at x=0 or x=100.
    if position_pct >= 85:
        xanchor = "right"
    elif position_pct <= 15:
        xanchor = "left"
    else:
        xanchor = "center"

    fig.add_annotation(
        x=position_pct, y=1.5, showarrow=False,
        text=f"<b>{label}</b>", font=dict(size=13),
        xanchor=xanchor,
    )

    # Small padding on the x-axis range so edge labels have room to render
    # even with center anchoring, and so the marker line isn't flush with the frame.
    fig.update_xaxes(range=[-8, 108], visible=False)
    fig.update_yaxes(range=[-0.5, 2], visible=False)
    fig.update_layout(
        height=110,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    return fig


def render_probability_bar(probs: np.ndarray, predicted_idx: int):
    """Horizontal bar chart of per-class probabilities, predicted class highlighted."""
    colors = ["#BA7517" if i == predicted_idx else "#d9d9d9" for i in range(len(CLASS_ORDER))]
    fig = go.Figure(go.Bar(
        x=probs * 100,
        y=CLASS_ORDER,
        orientation="h",
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside",
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis=dict(title="Probability (%)", range=[0, 100]),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def render_bmi_reference_table(predicted_label: str) -> str:
    """Single self-contained HTML table (safe to nest since it's one st.markdown call)."""
    rows = ""
    for label in CLASS_ORDER:
        row_class = "highlight" if label == predicted_label else ""
        rows += f'<tr class="{row_class}"><td>{label}</td><td>{BMI_RANGES[label]}</td></tr>'
    return f"""
    <table class="bmi-table">
        <tr><th>Category</th><th>BMI range (kg/m²)</th></tr>
        {rows}
    </table>
    """


def display_prediction_results(prediction_index: int, accuracy_score: float, probabilities):
    """Shared results UI for any model: gauge, confidence, probability chart, BMI reference.
    Used by both the ANN and KNN branches so the two models render identically."""
    predicted_label = target_reverse_map.get(prediction_index, "Unknown Class")
    position_pct = (prediction_index / (len(CLASS_ORDER) - 1)) * 100

    # st.container(border=True) actually wraps everything placed inside it,
    # unlike two separate st.markdown('<div>') / st.markdown('</div>') calls.
    with st.container(border=True):

        top_col1, top_col2, top_col3 = st.columns(3)
        with top_col1:
            st.markdown('<p class="result-label">Predicted category</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="result-value">{predicted_label}</p>', unsafe_allow_html=True)
        with top_col2:
            st.markdown('<p class="result-label">Model benchmark accuracy</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="result-value">{accuracy_score:.1f}%</p>', unsafe_allow_html=True)
        with top_col3:
            st.markdown('<p class="result-label">Calculated BMI</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="result-value">{raw_input_data["BMI"]}</p>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.plotly_chart(render_gauge(position_pct, predicted_label), use_container_width=True)

        if probabilities is not None:
            st.markdown('<p class="result-label">Probability by class</p>', unsafe_allow_html=True)
            st.plotly_chart(
                render_probability_bar(np.asarray(probabilities), prediction_index),
                use_container_width=True
            )
        else:
            st.markdown(
                '<p class="small-note">Per-class probability breakdown unavailable for this model.</p>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("📖 Understanding this result — BMI classification reference"):
            st.markdown(render_bmi_reference_table(predicted_label), unsafe_allow_html=True)
            st.markdown(
                '<p class="small-note" style="margin-top:8px;">'
                'BMI = weight (kg) ÷ height (m)². These ranges follow the standard '
                'classification used in the model\'s training dataset and are for '
                'general reference, not a medical diagnosis.</p>',
                unsafe_allow_html=True
            )


@st.cache_resource
def get_trained_knn_model():
    """Trains KNN once per session instead of on every button click."""
    return train_knn_model()


# --- 5. PREDICTION TRIGGERS ---
st.markdown("---")
if st.button("Run", type="primary"):

    if selected_model == "ANN (Artificial Neural Network)":
        with st.spinner("Processing Artificial Neural Network..."):
            try:
                result = predict_ann(input_df)

                probabilities = None
                if isinstance(result, tuple) and len(result) == 3:
                    prediction_index, accuracy_score, probabilities = result
                else:
                    prediction_index, accuracy_score = result

                display_prediction_results(prediction_index, accuracy_score, probabilities)

            except Exception as e:
                st.error(f"An unexpected computational error occurred: {e}")

    elif selected_model == "KNN (K-Nearest Neighbors)":
        with st.spinner("Processing K-Nearest Neighbors..."):
            try:
                model, scaler, expected_columns_order, accuracy_score = get_trained_knn_model()
                result = predict_knn(model, scaler, expected_columns_order, input_df)

                probabilities = None
                if isinstance(result, tuple) and len(result) == 2:
                    prediction_index, probabilities = result
                else:
                    prediction_index = result

                display_prediction_results(prediction_index, accuracy_score, probabilities)

            except Exception as e:
                st.error(f"An unexpected computational error occurred: {e}")

    else:
        st.warning(f"Functional pipelines for **{selected_model}** are currently preparation blueprints. Please select **ANN** or **KNN** to run predictions.")