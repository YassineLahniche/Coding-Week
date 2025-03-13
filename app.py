import streamlit as st
import pickle
import numpy as np
import xgboost as xgb
import requests
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import shap 
from streamlit_shap import st_shap

# Set page configuration
st.set_page_config(
    page_title="Obesity Risk Prediction",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Apply enhanced custom CSS with modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background-color: #f5f7fa;
    }
    
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    
    .main-subheader {
        font-size: 1.1rem;
        font-weight: 400;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .subheader {
        font-size: 1.6rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1.2rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #334155;
        margin: 1.2rem 0 0.8rem 0;
        padding-bottom: 0.2rem;
    }
    
    .card {
        background-color: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #f1f5f9;
        transition: all 0.2s ease;
    }
    
    .card:hover {
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        border-color: #e2e8f0;
    }
    
    .metric-container {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    
    .metric-item {
        text-align: center;
        padding: 1rem;
        min-width: 140px;
        border-radius: 8px;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        transition: transform 0.2s ease;
    }
    
    .metric-item:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        line-height: 1.1;
    }
    
    .metric-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: #64748b;
    }
    
    .prediction-result {
        padding: 24px;
        border-radius: 10px;
        text-align: center;
        margin: 1.5rem 0;
        transition: all 0.3s ease;
    }
    
    .prediction-result h2 {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        opacity: 0.9;
    }
    
    .prediction-result h1 {
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .tab-content {
        padding: 1.2rem;
        background-color: #f8fafc;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        border: none;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.25);
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: #2563eb;
        box-shadow: 0 6px 10px rgba(59, 130, 246, 0.3);
        transform: translateY(-2px);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    .explanation-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 1rem;
        border-left: 4px solid #3b82f6;
    }
    
    .feature-impact {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
        padding: 8px 12px;
        border-radius: 6px;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    
    .feature-name {
        font-weight: 600;
        flex: 1;
    }
    
    .feature-value {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        background-color: #f1f5f9;
        margin: 0 8px;
    }
    
    .impact-indicator {
        display: flex;
        align-items: center;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    .impact-positive {
        color: #ef4444;
    }
    
    .impact-negative {
        color: #10b981;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        border: 1px solid #e2e8f0;
        border-bottom: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        border-color: #e2e8f0;
        border-bottom: none;
        font-weight: 600;
    }
    
    .placeholder-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem;
        color: #94a3b8;
        background-color: #f8fafc;
        border-radius: 12px;
        border: 2px dashed #e2e8f0;
    }
    
    .placeholder-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.4;
    }
    
    .placeholder-text {
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    /* Form styling */
    div[data-baseweb="select"] {
        border-radius: 8px;
    }
    
    div[data-baseweb="select"] > div {
        background-color: white;
        border-radius: 8px;
        border-color: #e2e8f0;
    }
    
    div[data-baseweb="base-input"] {
        border-radius: 8px;
    }
    
    div[data-baseweb="base-input"] > div {
        background-color: white;
        border-radius: 8px;
        border-color: #e2e8f0;
    }
    
    /* Custom section divider */
    .section-divider {
        display: flex;
        align-items: center;
        margin: 1.5rem 0;
    }
    
    .section-divider-line {
        flex: 1;
        height: 1px;
        background-color: #e2e8f0;
    }
    
    .section-divider-text {
        font-size: 0.9rem;
        font-weight: 600;
        color: #64748b;
        padding: 0 12px;
    }
    
    /* SHAP plots styling */
    .shap-plot-container {
        background-color: white;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e2e8f0;
        margin-top: 1rem;
    }
    
    .shap-plot-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #334155;
        margin-bottom: 0.5rem;
    }
    
    .shap-plot-description {
        font-size: 0.9rem;
        color: #64748b;
        margin-bottom: 1rem;
    }
    
    /* Additional visualization enhancements */
    .category-label {
        font-size: 1rem;
        font-weight: 600;
        text-align: center;
        padding: 8px;
        border-radius: 6px;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# Load model function
def load_model():
    try:
        # Step 1: Define the URL to the raw model file in the GitHub repository
        model_url = "https://github.com/YassineLahniche/Coding-Week/raw/main/model/xgb_baseline.pkl"

        # Step 2: Download the model file
        response = requests.get(model_url)

        # Check if the request was successful
        if response.status_code == 200:
            # Step 3: Load the model using pickle
            model = pickle.loads(response.content)
            return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def calculate_bmi(weight, height):
    height_m = height / 100  # Convert cm to meters
    return round(weight / (height_m ** 2), 2)

def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight", "#3b82f6"  # Blue
    elif 18.5 <= bmi < 25:
        return "Normal weight", "#10b981"  # Green
    elif 25 <= bmi < 30:
        return "Overweight", "#f59e0b"  # Amber
    elif 30 <= bmi < 35:
        return "Obesity Class I", "#f97316"  # Orange
    elif 35 <= bmi < 40:
        return "Obesity Class II", "#ef4444"  # Red
    else:
        return "Obesity Class III", "#b91c1c"  # Dark red

def predict_obesity_level(model, input_data):
    obesity_categories = {
        0: 'Insufficient Weight',
        1: 'Normal Weight',
        2: 'Overweight Level I',
        3: 'Overweight Level II',
        4: 'Obesity Type I',
        5: 'Obesity Type II',
        6: 'Obesity Type III'
    }
    
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    category = obesity_categories.get(int(prediction), "Unknown")
    
    return prediction, category

def get_category_color(category):
    colors = {
        'Insufficient Weight': '#3b82f6',  # Blue
        'Normal Weight': '#10b981',        # Green
        'Overweight Level I': '#f59e0b',   # Amber
        'Overweight Level II': '#f97316',  # Orange
        'Obesity Type I': '#ef4444',       # Red
        'Obesity Type II': '#dc2626',      # Darker red
        'Obesity Type III': '#b91c1c'      # Darkest red
    }
    return colors.get(category, "#94a3b8")  # Default slate gray

def create_bmi_gauge(bmi):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = bmi,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "BMI", 'font': {'size': 24, 'family': 'Poppins, sans-serif', 'color': '#1e293b'}},
        gauge = {
            'axis': {'range': [None, 45], 'tickwidth': 1, 'tickcolor': "#64748b", 'tickfont': {'family': 'Poppins, sans-serif'}},
            'bar': {'color': "#3b82f6"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 18.5], 'color': '#3b82f6'},   # Blue
                {'range': [18.5, 25], 'color': '#10b981'},  # Green
                {'range': [25, 30], 'color': '#f59e0b'},    # Amber
                {'range': [30, 35], 'color': '#f97316'},    # Orange
                {'range': [35, 40], 'color': '#ef4444'},    # Red
                {'range': [40, 45], 'color': '#b91c1c'}     # Dark red
            ],
            'threshold': {
                'line': {'color': "#1e293b", 'width': 4},
                'thickness': 0.75,
                'value': bmi
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#1e293b", 'family': "Poppins, sans-serif"}
    )
    return fig

def create_category_chart(category):
    categories = ['Insufficient Weight', 'Normal Weight', 'Overweight Level I', 
                 'Overweight Level II', 'Obesity Type I', 'Obesity Type II', 'Obesity Type III']
    
    # Create a horizontal bar chart
    fig = go.Figure()
    
    # Add colored bars for each category
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#f97316', '#ef4444', '#dc2626', '#b91c1c']
    
    for i, cat in enumerate(categories):
        opacity = 0.3
        if cat == category:
            opacity = 1.0
        
        fig.add_trace(go.Bar(
            y=[cat],
            x=[1],
            orientation='h',
            marker=dict(color=colors[i], opacity=opacity),
            showlegend=False,
            hoverinfo='none'
        ))
    
    # Add a marker for the predicted category
    category_index = categories.index(category)
    
    fig.add_annotation(
        x=0.5,
        y=category_index,
        text="✓",
        showarrow=False,
        font=dict(size=24, color="white")
    )
    
    fig.update_layout(
        height=300,
        barmode='stack',
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(autorange="reversed", tickfont={'family': "Poppins, sans-serif", 'color': '#1e293b'}),
        font={'color': "#1e293b", 'family': "Poppins, sans-serif"}
    )
    
    return fig

def create_shap_explainer(model):
    explainer = shap.TreeExplainer(model)
    return explainer

# Calculate SHAP values for a specific instance
def calculate_shap_values(explainer, instance):
    shap_values = explainer.shap_values(instance)
    return shap_values

# Create SHAP force plot
def create_shap_force_plot(explainer, shap_values, instance, feature_names):
    plt.figure(figsize=(10, 3))
    
    if hasattr(explainer, 'expected_value'):
        if isinstance(explainer.expected_value, list):
            expected_value = explainer.expected_value[0]
        else:
            expected_value = explainer.expected_value
    else:
        expected_value = 0
    
    # Fix: Remove `matplotlib=True`
    shap_plot = shap.force_plot(expected_value, shap_values, instance, feature_names=feature_names)
    return shap_plot


# Create SHAP waterfall plot
def create_shap_waterfall_plot(explainer, shap_values, instance, feature_names):
    plt.figure(figsize=(10, 6))
    
    # Get the expected value (base value)
    if hasattr(explainer, 'expected_value'):
        if isinstance(explainer.expected_value, list):
            # For multi-class models
            expected_value = explainer.expected_value[0]
        else:
            # For binary classification or regression
            expected_value = explainer.expected_value
    else:
        # If no expected_value attribute, use 0 as fallback
        expected_value = 0
    
    shap.plots._waterfall.waterfall_legacy(expected_value, shap_values, 
                                          feature_names=feature_names, show=False)
    plt.tight_layout()
    return plt

# Create SHAP bar plot for feature importance
def create_shap_bar_plot(explainer, input_array, feature_names):
    plt.figure(figsize=(10, 6))
    shap_values = explainer.shap_values(input_array)
    shap.summary_plot(shap_values, input_array, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    return plt

# Create SHAP decision plot
def create_shap_decision_plot(explainer, shap_values, instance, feature_names):
    plt.figure(figsize=(10, 8))
    
    # Get the expected value (base value)
    if hasattr(explainer, 'expected_value'):
        if isinstance(explainer.expected_value, list):
            # For multi-class models
            expected_value = explainer.expected_value[0]
        else:
            # For binary classification or regression
            expected_value = explainer.expected_value
    else:
        # If no expected_value attribute, use 0 as fallback
        expected_value = 0
    
    shap.decision_plot(expected_value, shap_values, feature_names=feature_names, show=False)
    plt.tight_layout()
    return plt

# Get feature names
def get_feature_names():
    return [
        "Gender", "Age", "Height", "Weight", "Family_History", "High_Caloric_Food",
        "Vegetable_Consumption", "Main_Meals", "Snacking", "Smoking",
        "Water_Intake", "Calorie_Monitoring", "Physical_Activity",
        "Screen_Time", "Alcohol", "Transportation"
    ]

# Translate encoded values to human-readable format
def get_readable_values(input_data):
    # Mapping dictionaries for encoded values
    gender_map = {0: "Male", 1: "Female"}
    yes_no_map = {0: "No", 1: "Yes"}
    veggie_map = {0: "Never", 1: "Sometimes", 2: "Always"}
    meals_map = {0: "1-2", 1: "3", 2: ">3"}
    frequency_map = {0: "Never", 1: "Sometimes", 2: "Frequently", 3: "Always"}
    water_map = {0: "Less than 1L", 1: "1-2L", 2: "More than 2L"}
    activity_map = {0: "Never", 1: "Once or twice a week", 2: "Two or three times a week", 3: "More than three times a week"}
    screen_map = {0: "None", 1: "Less than 1h", 2: "1-3h", 3: "More than 3h"}
    transport_map = {0: "Automobile", 1: "Public Transportation", 2: "Motorbike", 3: "Bike", 4: "Walking"}
    
    # Extract values from input_data
    gender = gender_map[input_data[0]]
    age = input_data[1]
    height = input_data[2]
    weight = input_data[3]
    family_history = yes_no_map[input_data[4]]
    high_caloric_food = yes_no_map[input_data[5]]
    veggie_freq = veggie_map[input_data[6]]
    main_meals = meals_map[input_data[7]]
    snacking = frequency_map[input_data[8]]
    smoking = yes_no_map[input_data[9]]
    water_intake = water_map[input_data[10]]
    calorie_monitoring = yes_no_map[input_data[11]]
    physical_activity = activity_map[input_data[12]]
    screen_time = screen_map[input_data[13]]
    alcohol = frequency_map[input_data[14]]
    transportation = transport_map[input_data[15]]
    
    return {
        "Gender": gender,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "Family_History": family_history,
        "High_Caloric_Food": high_caloric_food,
        "Vegetable_Consumption": veggie_freq,
        "Main_Meals": main_meals,
        "Snacking": snacking,
        "Smoking": smoking,
        "Water_Intake": water_intake,
        "Calorie_Monitoring": calorie_monitoring,
        "Physical_Activity": physical_activity,
        "Screen_Time": screen_time,
        "Alcohol": alcohol,
        "Transportation": transportation
    }

# Generate text explanation based on SHAP values
def generate_text_explanation(shap_values, feature_names, readable_values):
    # Convert shap_values to a 1D array if it's not already
    if len(shap_values.shape) > 1:
        shap_values = shap_values.flatten()
    
    # Ensure we only use valid indices
    valid_indices = [i for i in range(len(shap_values)) if i < len(feature_names)]
    
    # Get indices of top features by absolute value (only from valid indices)
    top_indices = sorted(valid_indices, key=lambda i: abs(shap_values[i]), reverse=True)[:5]
    
    # Create explanation html
    explanation = """
    <div class="explanation-card">
        <h3 style="margin-top: 0; font-size: 1.2rem; font-weight: 600; color: #1e293b; margin-bottom: 1rem;">Key Factors Influencing This Prediction</h3>
    """
    
    for idx in top_indices:
        feature = feature_names[idx]
        display_feature = feature.replace('_', ' ')
        value = readable_values.get(feature, "Unknown")
        impact = float(shap_values[idx])
        impact_abs = abs(impact)
        
        if impact > 0:
            direction = "increases"
            color = "#ef4444"  # Red for increasing risk
            icon = "↑"
        else:   
            direction = "decreases"
            color = "#10b981"  # Green for decreasing risk
            icon = "↓"
        
        explanation += f"""
        <div class="feature-impact">
            <div class="feature-name">{display_feature}</div>
            <div class="feature-value">{value}</div>
            <div class="impact-indicator" style="color: {color}">
                {icon} {direction} risk {impact_abs:.2f}
            </div>
        </div>
        """
    
    explanation += "</div>"
    
    return explanation

# Main application header
st.markdown('<h1 class="main-header">Obesity Risk Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subheader">Assess your risk factors and get personalized health insights</p>', unsafe_allow_html=True)

# Load model
model = load_model()

# Create tabs for different app sections
tabs = st.tabs(["Risk Assessment", "Health Profile", "Model Insights"])

with tabs[0]:
    st.markdown('<h2 class="subheader">Personal Risk Assessment</h2>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Create two columns for input form
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<p class="section-title">Demographics</p>', unsafe_allow_html=True)
            gender = st.selectbox("Gender", options=["Male", "Female"], index=0)
            age = st.number_input("Age", min_value=10, max_value=100, value=30)
            height = st.number_input("Height (cm)", min_value=120, max_value=220, value=170)
            weight = st.number_input("Weight (kg)", min_value=30, max_value=250, value=70)
            
            st.markdown('<p class="section-title">Health History</p>', unsafe_allow_html=True)
            family_history = st.selectbox("Family History of Obesity", options=["No", "Yes"], index=0)
            smoking = st.selectbox("Do you smoke?", options=["No", "Yes"], index=0)
            alcohol = st.selectbox("Alcohol Consumption", options=["Never", "Sometimes", "Frequently", "Always"], index=1)
        
        with col2:
            st.markdown('<p class="section-title">Dietary Habits</p>', unsafe_allow_html=True)
            high_caloric_food = st.selectbox("High Caloric Food Consumption", options=["No", "Yes"], index=0)
            vegetable_consumption = st.selectbox("Vegetable Consumption", options=["Never", "Sometimes", "Always"], index=1)
            main_meals = st.selectbox("Number of Main Meals", options=["1-2", "3", ">3"], index=1)
            snacking = st.selectbox("Frequency of Snacking", options=["Never", "Sometimes", "Frequently", "Always"], index=1)
            water_intake = st.selectbox("Daily Water Intake", options=["Less than 1L", "1-2L", "More than 2L"], index=1)
            calorie_monitoring = st.selectbox("Do you monitor calories?", options=["No", "Yes"], index=0)
            
            st.markdown('<p class="section-title">Lifestyle</p>', unsafe_allow_html=True)
            physical_activity = st.selectbox("Physical Activity Frequency", 
                                          options=["Never", "Once or twice a week", "Two or three times a week", "More than three times a week"], 
                                          index=1)
            screen_time = st.selectbox("Daily Screen Time", options=["None", "Less than 1h", "1-3h", "More than 3h"], index=2)
            transportation = st.selectbox("Primary Mode of Transportation", 
                                      options=["Automobile", "Public Transportation", "Motorbike", "Bike", "Walking"], 
                                      index=0)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Create a button to predict
        predict_button = st.button("Predict Obesity Risk", type="primary")
        
        # Process prediction when button is clicked
        if predict_button and model:
            # Transform inputs into model format
            gender_encoded = 0 if gender == "Male" else 1
            family_history_encoded = 0 if family_history == "No" else 1
            high_caloric_food_encoded = 0 if high_caloric_food == "No" else 1
            vegetable_map = {"Never": 0, "Sometimes": 1, "Always": 2}
            vegetable_encoded = vegetable_map[vegetable_consumption]
            meals_map = {"1-2": 0, "3": 1, ">3": 2}
            meals_encoded = meals_map[main_meals]
            frequency_map = {"Never": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
            snacking_encoded = frequency_map[snacking]
            smoking_encoded = 0 if smoking == "No" else 1
            water_map = {"Less than 1L": 0, "1-2L": 1, "More than 2L": 2}
            water_encoded = water_map[water_intake]
            calorie_monitoring_encoded = 0 if calorie_monitoring == "No" else 1
            activity_map = {"Never": 0, "Once or twice a week": 1, "Two or three times a week": 2, "More than three times a week": 3}
            activity_encoded = activity_map[physical_activity]
            screen_map = {"None": 0, "Less than 1h": 1, "1-3h": 2, "More than 3h": 3}
            screen_encoded = screen_map[screen_time]
            alcohol_encoded = frequency_map[alcohol]
            transport_map = {"Automobile": 0, "Public Transportation": 1, "Motorbike": 2, "Bike": 3, "Walking": 4}
            transport_encoded = transport_map[transportation]
            
            # Create input array for model
            input_data = [
                gender_encoded, age, height, weight, family_history_encoded, 
                high_caloric_food_encoded, vegetable_encoded, meals_encoded, 
                snacking_encoded, smoking_encoded, water_encoded, calorie_monitoring_encoded, 
                activity_encoded, screen_encoded, alcohol_encoded, transport_encoded
            ]
            
            # Predict obesity category
            prediction, category = predict_obesity_level(model, input_data)
            
            # Calculate BMI
            bmi = calculate_bmi(weight, height)
            bmi_category, bmi_color = get_bmi_category(bmi)
            
            # Display results
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            # Create two columns for results
            results_col1, results_col2 = st.columns(2)
            
            with results_col1:
                st.markdown(f'<h3 class="section-title">Predicted Obesity Category</h3>', unsafe_allow_html=True)
                category_color = get_category_color(category)
                st.markdown(f'<div class="prediction-result" style="background-color: {category_color}20; border: 1px solid {category_color};">'
                          f'<h2 style="color: {category_color}">Prediction Result</h2>'
                          f'<h1 style="color: {category_color}">{category}</h1>'
                          f'</div>', unsafe_allow_html=True)
                
                # Display category chart
                category_chart = create_category_chart(category)
                st.plotly_chart(category_chart, use_container_width=True)
                
            with results_col2:
                st.markdown(f'<h3 class="section-title">Body Mass Index (BMI)</h3>', unsafe_allow_html=True)
                st.markdown(f'<div class="prediction-result" style="background-color: {bmi_color}20; border: 1px solid {bmi_color};">'
                          f'<h2 style="color: {bmi_color}">BMI Category</h2>'
                          f'<h1 style="color: {bmi_color}">{bmi_category} ({bmi})</h1>'
                          f'</div>', unsafe_allow_html=True)
                
                # Display BMI gauge
                bmi_gauge = create_bmi_gauge(bmi)
                st.plotly_chart(bmi_gauge, use_container_width=True)
            
            # Add a divider
            st.markdown('<div class="section-divider"><div class="section-divider-line"></div>'
                      '<div class="section-divider-text">Risk Factors Analysis</div>'
                      '<div class="section-divider-line"></div></div>', unsafe_allow_html=True)
            
            # SHAP Analysis
            if model:
                try:
                    # Create SHAP explainer
                    input_array = np.array(input_data).reshape(1, -1)
                    explainer = create_shap_explainer(model)
                    
                    # Calculate SHAP values for this instance
                    shap_values = calculate_shap_values(explainer, input_array)
                    shap_values = np.array(shap_values)

                    if isinstance(explainer.expected_value, list) and shap_values.ndim > 1:
                        shap_values = shap_values[0]  


                    
                    # Get feature names and readable values
                    feature_names = get_feature_names()
                    readable_values = get_readable_values(input_data)
                    
                    # Generate text explanation
                    explanation_html = generate_text_explanation(shap_values, feature_names, readable_values)
                    st.markdown(explanation_html, unsafe_allow_html=True)
                    
                    # Create SHAP plots
                    shap_col1, shap_col2 = st.columns(2)
                    
                    with shap_col1:
                        st.markdown('<div class="shap-plot-container">', unsafe_allow_html=True)
                        st.markdown('<p class="shap-plot-title">Feature Impact on Prediction</p>', unsafe_allow_html=True)
                        st.markdown('<p class="shap-plot-description">How each factor pushes the prediction higher (red) or lower (blue)</p>', unsafe_allow_html=True)
                        
                        force_plot = create_shap_force_plot(explainer, shap_values, input_array[0], feature_names)
                        st_shap(force_plot, height=300)  # Use `st_shap` to render in Streamlit
                        
                        st.markdown('</div>', unsafe_allow_html=True)

                    
                    with shap_col2:
                        st.markdown('<div class="shap-plot-container">', unsafe_allow_html=True)
                        st.markdown('<p class="shap-plot-title">Waterfall Plot</p>', unsafe_allow_html=True)
                        st.markdown('<p class="shap-plot-description">How each factor contributes to the final prediction</p>', unsafe_allow_html=True)
                        waterfall_plot = create_shap_waterfall_plot(explainer, shap_values, input_array[0], feature_names)
                        st.pyplot(waterfall_plot, clear_figure=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error generating SHAP analysis: {e}")
                    
            st.markdown('</div>', unsafe_allow_html=True)

with tabs[1]:
    st.markdown('<h2 class="subheader">Your Health Profile</h2>', unsafe_allow_html=True)
    
    if 'prediction' not in locals():
        st.markdown('<div class="placeholder-content">'
                  '<div class="placeholder-icon">📊</div>'
                  '<div class="placeholder-text">Complete the Risk Assessment first to view your Health Profile</div>'
                  '</div>', unsafe_allow_html=True)
    else:
        # Display health metrics
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">Key Health Metrics</h3>', unsafe_allow_html=True)
        
        # Create metric containers
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        
        # BMI metric
        st.markdown(f'<div class="metric-item">'
                  f'<div class="metric-value" style="color: {bmi_color}">{bmi}</div>'
                  f'<div class="metric-label">BMI</div>'
                  f'</div>', unsafe_allow_html=True)
        
        # Age metric
        st.markdown(f'<div class="metric-item">'
                  f'<div class="metric-value" style="color: #3b82f6">{age}</div>'
                  f'<div class="metric-label">Age</div>'
                  f'</div>', unsafe_allow_html=True)
        
        # Height metric
        st.markdown(f'<div class="metric-item">'
                  f'<div class="metric-value" style="color: #8b5cf6">{height} cm</div>'
                  f'<div class="metric-label">Height</div>'
                  f'</div>', unsafe_allow_html=True)
        
        # Weight metric
        st.markdown(f'<div class="metric-item">'
                  f'<div class="metric-value" style="color: #ec4899">{weight} kg</div>'
                  f'<div class="metric-label">Weight</div>'
                  f'</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Lifestyle summary
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">Lifestyle Summary</h3>', unsafe_allow_html=True)
        
        # Create columns for lifestyle factors
        ls_col1, ls_col2, ls_col3 = st.columns(3)
        
        with ls_col1:
            st.markdown('<p class="section-title">Physical Activity</p>', unsafe_allow_html=True)
            activity_color = "#ef4444" if activity_encoded < 2 else "#10b981"
            st.markdown(f'<div style="padding: 16px; border-radius: 8px; background-color: {activity_color}20; border: 1px solid {activity_color};">'
                      f'<div style="font-weight: 600; font-size: 1.1rem; color: {activity_color}">{physical_activity}</div>'
                      f'<div style="font-size: 0.9rem; color: #64748b; margin-top: 4px;">Activity Level</div>'
                      f'</div>', unsafe_allow_html=True)
        
        with ls_col2:
            st.markdown('<p class="section-title">Diet Quality</p>', unsafe_allow_html=True)
            diet_score = vegetable_encoded * 2 - (1 if high_caloric_food_encoded else 0) + (1 if calorie_monitoring_encoded else 0)
            diet_status = "Needs Improvement" if diet_score < 2 else "Good" if diet_score < 4 else "Excellent"
            diet_color = "#ef4444" if diet_score < 2 else "#f59e0b" if diet_score < 4 else "#10b981"
            
            st.markdown(f'<div style="padding: 16px; border-radius: 8px; background-color: {diet_color}20; border: 1px solid {diet_color};">'
                      f'<div style="font-weight: 600; font-size: 1.1rem; color: {diet_color}">{diet_status}</div>'
                      f'<div style="font-size: 0.9rem; color: #64748b; margin-top: 4px;">Diet Quality</div>'
                      f'</div>', unsafe_allow_html=True)
        
        with ls_col3:
            st.markdown('<p class="section-title">Risk Factors</p>', unsafe_allow_html=True)
            risk_count = family_history_encoded + smoking_encoded + (1 if alcohol_encoded > 1 else 0) + (1 if screen_encoded > 2 else 0)
            risk_status = "Low" if risk_count < 1 else "Moderate" if risk_count < 3 else "High"
            risk_color = "#10b981" if risk_count < 1 else "#f59e0b" if risk_count < 3 else "#ef4444"
            
            st.markdown(f'<div style="padding: 16px; border-radius: 8px; background-color: {risk_color}20; border: 1px solid {risk_color};">'
                      f'<div style="font-weight: 600; font-size: 1.1rem; color: {risk_color}">{risk_status}</div>'
                      f'<div style="font-size: 0.9rem; color: #64748b; margin-top: 4px;">Risk Level</div>'
                      f'</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Recommendations
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">Personalized Recommendations</h3>', unsafe_allow_html=True)
        
        recommendations = []
        
        # Diet recommendations
        if vegetable_encoded < 2:
            recommendations.append("Increase your vegetable consumption to at least 5 servings per day")
        
        if high_caloric_food_encoded:
            recommendations.append("Reduce consumption of high-calorie processed foods")
        
        if water_encoded < 2:
            recommendations.append("Increase water intake to more than 2 liters per day")
        
        if not calorie_monitoring_encoded:
            recommendations.append("Consider monitoring your calorie intake to better understand your diet")
        
        # Activity recommendations
        if activity_encoded < 2:
            recommendations.append("Increase physical activity to at least 2-3 times per week")
        
        if screen_encoded > 2:
            recommendations.append("Reduce daily screen time to less than 2 hours")
        
        if transport_encoded < 3:
            recommendations.append("Consider active transportation like biking or walking when possible")
        
        # Health risk recommendations
        if smoking_encoded:
            recommendations.append("Quitting smoking will significantly improve your health")
        
        if alcohol_encoded > 1:
            recommendations.append("Reduce alcohol consumption")
        
        if family_history_encoded:
            recommendations.append("With family history of obesity, regular health checkups are recommended")
        
        # Add general recommendations if few specific ones
        if len(recommendations) < 3:
            recommendations.append("Maintain a balanced diet rich in fruits, vegetables, and lean proteins")
            recommendations.append("Aim for at least 150 minutes of moderate activity per week")
            recommendations.append("Ensure adequate sleep of 7-9 hours each night")
        
        # Display recommendations
        for i, rec in enumerate(recommendations[:6]):  # Limit to 6 recommendations
            st.markdown(f'<div style="padding: 12px; border-radius: 8px; background-color: #f8fafc; margin-bottom: 8px; border-left: 4px solid #3b82f6;">'
                      f'<div style="font-weight: 500;">{i+1}. {rec}</div>'
                      f'</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

with tabs[2]:
    st.markdown('<h2 class="subheader">Model Insights</h2>', unsafe_allow_html=True)
    
    # Model information
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">About the Prediction Model</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    <p style="margin-bottom: 16px;">This application uses a machine learning model trained on health and lifestyle data to predict obesity risk categories.</p>
    
    <p style="margin-bottom: 16px;">The model analyzes 16 different factors including demographics, diet habits, physical activity, and lifestyle choices to classify individuals into one of seven weight categories.</p>
    
    <div style="padding: 16px; border-radius: 8px; background-color: #f8fafc; margin-bottom: 16px;">
        <p style="font-weight: 600; margin-bottom: 8px;">Weight Categories:</p>
        <ul style="margin-left: 20px; margin-bottom: 0;">
            <li><span style="color: #3b82f6; font-weight: 500;">Insufficient Weight</span> - BMI below 18.5</li>
            <li><span style="color: #10b981; font-weight: 500;">Normal Weight</span> - BMI between 18.5 and 24.9</li>
            <li><span style="color: #f59e0b; font-weight: 500;">Overweight Level I</span> - BMI between 25 and 27.49</li>
            <li><span style="color: #f97316; font-weight: 500;">Overweight Level II</span> - BMI between 27.5 and 29.9</li>
            <li><span style="color: #ef4444; font-weight: 500;">Obesity Type I</span> - BMI between 30 and 34.9</li>
            <li><span style="color: #dc2626; font-weight: 500;">Obesity Type II</span> - BMI between 35 and 39.9</li>
            <li><span style="color: #b91c1c; font-weight: 500;">Obesity Type III</span> - BMI 40 or higher</li>
        </ul>
    </div>
    
    <p>The model uses XGBoost, a powerful gradient boosting algorithm that excels at classification tasks. It has been trained on a dataset of individuals with various health profiles and validated to ensure accuracy.</p>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Important factors visualization
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">Key Factors in Obesity Risk</h3>', unsafe_allow_html=True)
    
    # Check if model is available
    if model:
        try:
            # Create a sample input for visualization
            sample_input = np.array([
                [0, 30, 170, 70, 0, 0, 1, 1, 1, 0, 1, 0, 1, 2, 1, 0]
            ])
            
            # Create SHAP explainer
            explainer = create_shap_explainer(model)
            feature_names = get_feature_names()
            
            # Create feature importance bar plot
            importance_plot = create_shap_bar_plot(explainer, sample_input, feature_names)
            st.pyplot(importance_plot, clear_figure=True)
            
            st.markdown("""
            <p style="margin-top: 16px;">This chart shows which factors have the most impact on obesity risk predictions. Factors at the top have the highest influence on the model's decisions. The SHAP value measures how much each factor contributes to pushing the prediction higher or lower from the baseline.</p>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error generating model insights: {e}")
    else:
        st.warning("Model not available. Unable to display feature importance.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tips section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">Understanding Your Results</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    <p style="margin-bottom: 12px;">This tool provides predictions based on statistical patterns, but each person's health journey is unique. Here are some tips for interpreting your results:</p>
    
    <ol style="margin-left: 16px;">
        <li><strong>BMI is just one indicator</strong> - While useful, BMI doesn't account for muscle mass, body composition, or individual health circumstances.</li>
        <li><strong>Focus on modifiable factors</strong> - Pay attention to lifestyle factors you can change, like diet, physical activity, and sleep habits.</li>
        <li><strong>Small changes add up</strong> - Even modest improvements in key areas can significantly reduce obesity risk over time.</li>
        <li><strong>Consult healthcare professionals</strong> - Use these insights as conversation starters with your doctor or nutritionist.</li>
    </ol>
    
    <div style="padding: 12px; border-radius: 8px; background-color: #f8fafc; margin-top: 16px; border-left: 4px solid #3b82f6;">
        <p style="margin: 0; font-style: italic;">This tool is for informational purposes and does not replace professional medical advice, diagnosis, or treatment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #e2e8f0;">
    <p style="color: #64748b; font-size: 0.9rem;">© 2025 Obesity Risk Assessment Tool | Created with Streamlit</p>
</div>
""", unsafe_allow_html=True)