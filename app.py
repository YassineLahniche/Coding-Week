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

# Set page configuration
st.set_page_config(
    page_title="Obesity Risk Prediction",
    page_icon="⚕️",
    layout="wide",
)

# Apply custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        font-weight: 500;
        color: #34495e;
        margin-bottom: 1rem;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
    }
    .metric-label {
        font-size: 1rem;
        text-align: center;
        color: #7f8c8d;
    }
    .prediction-result {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
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
        return "Underweight", "#3498db"
    elif 18.5 <= bmi < 25:
        return "Normal weight", "#2ecc71"
    elif 25 <= bmi < 30:
        return "Overweight", "#f39c12"
    elif 30 <= bmi < 35:
        return "Obesity Class I", "#e74c3c"
    elif 35 <= bmi < 40:
        return "Obesity Class II", "#c0392b"
    else:
        return "Obesity Class III", "#7b241c"

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
        'Insufficient Weight': '#3498db',
        'Normal Weight': '#2ecc71',
        'Overweight Level I': '#f39c12',
        'Overweight Level II': '#e67e22',
        'Obesity Type I': '#e74c3c',
        'Obesity Type II': '#c0392b',
        'Obesity Type III': '#7b241c'
    }
    return colors.get(category, "#95a5a6")

def create_bmi_gauge(bmi):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = bmi,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "BMI", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [None, 45], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 18.5], 'color': '#3498db'},
                {'range': [18.5, 25], 'color': '#2ecc71'},
                {'range': [25, 30], 'color': '#f39c12'},
                {'range': [30, 35], 'color': '#e74c3c'},
                {'range': [35, 40], 'color': '#c0392b'},
                {'range': [40, 45], 'color': '#7b241c'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': bmi
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#2c3e50", 'family': "Arial"}
    )
    return fig

def create_category_chart(category):
    categories = ['Insufficient Weight', 'Normal Weight', 'Overweight Level I', 
                 'Overweight Level II', 'Obesity Type I', 'Obesity Type II', 'Obesity Type III']
    
    # Create a horizontal bar chart
    fig = go.Figure()
    
    # Add colored bars for each category
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e67e22', '#e74c3c', '#c0392b', '#7b241c']
    
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
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(autorange="reversed"),
        font={'color': "#2c3e50", 'family': "Arial"}
    )
    
    return fig

def create_shap_explainer(model):
    explainer = shap.TreeExplainer(model)
    return explainer

# New: Calculate SHAP values for a specific instance
def calculate_shap_values(explainer, instance):
    shap_values = explainer.shap_values(instance)
    return shap_values

# New: Create SHAP force plot
def create_shap_force_plot(explainer, shap_values, instance, feature_names):
    plt.figure(figsize=(10, 3))
    
    # Get the expected value (base value)
    if hasattr(explainer, 'expected_value'):
        if isinstance(explainer.expected_value, list):
            expected_value = explainer.expected_value[0]
        else:
            expected_value = explainer.expected_value
    else:
        expected_value = 0
    
    # Ensure instance and shap_values have the same length
    if len(shap_values) != len(instance):
        # Resize to match - take the minimum length
        min_len = min(len(shap_values), len(instance))
        shap_values = shap_values[:min_len]
        instance = instance[:min_len]
    
    # Also ensure they match feature_names length
    if len(shap_values) != len(feature_names):
        # Resize feature_names or shap_values to match
        if len(shap_values) < len(feature_names):
            feature_names = feature_names[:len(shap_values)]
        else:
            shap_values = shap_values[:len(feature_names)]
    
    # Now instance and shap_values and feature_names should have the same length
    shap.force_plot(expected_value, shap_values, instance, feature_names=feature_names, 
                   matplotlib=True, show=False)
    plt.tight_layout()
    return plt

# New: Create SHAP waterfall plot
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

# New: Create SHAP bar plot for feature importance
def create_shap_bar_plot(explainer, input_array, feature_names):
    plt.figure(figsize=(10, 6))
    shap_values = explainer.shap_values(input_array)
    shap.summary_plot(shap_values, input_array, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    return plt

# New: Create SHAP decision plot
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

# New: Get feature names
def get_feature_names():
    return [
        "Gender", "Age", "Height", "Weight", "Family_History", "High_Caloric_Food",
        "Vegetable_Consumption", "Main_Meals", "Snacking", "Smoking",
        "Water_Intake", "Calorie_Monitoring", "Physical_Activity",
        "Screen_Time", "Alcohol", "Transportation"
    ]

# New: Translate encoded values to human-readable format
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
        "Family History": family_history,
        "High Caloric Food": high_caloric_food,
        "Vegetable Consumption": veggie_freq,
        "Main Meals": main_meals,
        "Snacking": snacking,
        "Smoking": smoking,
        "Water Intake": water_intake,
        "Calorie Monitoring": calorie_monitoring,
        "Physical Activity": physical_activity,
        "Screen Time": screen_time,
        "Alcohol": alcohol,
        "Transportation": transportation
    }

# New: Generate text explanation based on SHAP values
def generate_text_explanation(shap_values, feature_names, readable_values):
    # Convert shap_values to a 1D array if it's not already
    if len(shap_values.shape) > 1:
        shap_values = shap_values.flatten()
    
    # Ensure we only use valid indices
    valid_indices = [i for i in range(len(shap_values)) if i < len(feature_names)]
    
    # Get indices of top features by absolute value (only from valid indices)
    top_indices = sorted(valid_indices, key=lambda i: abs(shap_values[i]), reverse=True)[:5]
    
    # Create explanation text
    explanation = "### Key Factors in This Prediction\n\n"
    
    for idx in top_indices:
        feature = feature_names[idx]
        value = readable_values.get(feature, "Unknown")
        impact = float(shap_values[idx])
        
        if impact > 0:
            direction = "increased"
            color = "#e74c3c"  # Red for increasing risk
        else:   
            direction = "decreased"
            color = "#2ecc71"  # Green for decreasing risk
        
        explanation += f"- **{feature}** ({value}): <span style='color:{color}'>{direction} risk</span> (impact: {abs(impact):.2f})\n\n"
    
    return explanation



# Header with animation
header_col1, header_col2, header_col3 = st.columns([1, 2, 1])
with header_col2:
    st.markdown('<h1 class="main-header">Obesity Risk Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center;">Enter patient details to predict obesity risk category</p>', unsafe_allow_html=True)

# Create two columns for input and results
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="subheader">Patient Information</h2>', unsafe_allow_html=True)
    
    # Basic information
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    
    # Physical measurements
    st.markdown("#### Physical Measurements")
    height = st.number_input("Height (cm)", min_value=50, max_value=250, step=1)
    weight = st.number_input("Weight (kg)", min_value=10, max_value=300, step=1)
    
    # Health history
    st.markdown("#### Health History")
    family_history = st.selectbox("Family History of Overweight", ["No", "Yes"])
    smoking = st.selectbox("Smoking Habit", ["No", "Yes"])
    
    # Eating habits
    st.markdown("#### Lifestyle & Diet")
    tabs = st.tabs(["Diet", "Activity", "Habits"])
    
    with tabs[0]:
        high_caloric_food = st.selectbox("High Caloric Food Consumption", ["No", "Yes"])
        veggie_freq = st.selectbox("Vegetable Consumption Frequency", ["Never", "Sometimes", "Always"])
        main_meals = st.selectbox("Number of Main Meals per Day", ["1-2", "3", ">3"])
        snacking = st.selectbox("Snacking Frequency", ["Never", "Sometimes", "Frequently", "Always"])
        water_intake = st.selectbox("Water Intake", ["Less than 1L", "1-2L", "More than 2L"])
        calorie_monitoring = st.selectbox("Calorie Monitoring", ["No", "Yes"])
    
    with tabs[1]:
        physical_activity = st.selectbox("Physical Activity Frequency", ["Never", "Once or twice a week", "Two or three times a week", "More than three times a week"])
        transportation = st.selectbox("Transportation Method", ["Automobile", "Public Transportation", "Motorbike", "Bike", "Walking"])
    
    with tabs[2]:
        screen_time = st.selectbox("Screen Time", ["None", "Less than 1h", "1-3h", "More than 3h"])
        alcohol = st.selectbox("Alcohol Consumption", ["Never", "Sometimes", "Frequently", "Always"])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Encode categorical data
    encoding_dict = {
        "Male": 0, "Female": 1,
        "Yes": 1, "No": 0,
        "Never": 0, "Sometimes": 1, "Always": 2,
        "1-2": 0, "3": 1, ">3": 2,
        "Always": 3, "Frequently": 2, "Sometimes": 1, "Never": 0,
        "Less than 1L": 0, "1-2L": 1, "More than 2L": 2,
        "Never": 0, "Once or twice a week": 1, "Two or three times a week": 2, "More than three times a week": 3,
        "None": 0, "Less than 1h": 1, "1-3h": 2, "More than 3h": 3,
        "Automobile": 0, "Public Transportation": 1, "Motorbike": 2, "Bike": 3, "Walking": 4
    }
    
    input_data = [
        encoding_dict[gender], age, height, weight, encoding_dict[family_history], encoding_dict[high_caloric_food],
        encoding_dict[veggie_freq], encoding_dict[main_meals], encoding_dict[snacking], encoding_dict[smoking],
        encoding_dict[water_intake], encoding_dict[calorie_monitoring], encoding_dict[physical_activity],
        encoding_dict[screen_time], encoding_dict[alcohol], encoding_dict[transportation]
    ]
    
    predict_button = st.button("Predict", type="primary", use_container_width=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="subheader">Prediction Results</h2>', unsafe_allow_html=True)
    
    if predict_button:
        with st.spinner("Analyzing data..."):
            # Load model and make prediction
            model = load_model()
            if model:
                # Calculate BMI
                bmi = calculate_bmi(weight, height)
                bmi_category, bmi_color = get_bmi_category(bmi)
                
                # Get obesity prediction
                prediction, category = predict_obesity_level(model, input_data)
                category_color = get_category_color(category)
                
                # Display results
                st.markdown(f"""
                <div class="prediction-result" style="background-color: {category_color}; color: white;">
                    <h2>Predicted Category</h2>
                    <h1>{category}</h1>
                </div>
                """, unsafe_allow_html=True)
                
                # Display BMI gauge
                st.markdown("#### BMI Measurement")
                st.plotly_chart(create_bmi_gauge(bmi), use_container_width=True)
                
                # Display category visualization
                st.markdown("#### Risk Category")
                st.plotly_chart(create_category_chart(category), use_container_width=True)
                
                # Display BMI info
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; margin-top: 20px;">
                    <div>
                        <div class="metric-value">{bmi}</div>
                        <div class="metric-label">BMI Value</div>
                    </div>
                    <div>
                        <div class="metric-value" style="color: {bmi_color};">{bmi_category}</div>
                        <div class="metric-label">BMI Category</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # ----- SHAP Analysis Section -----
                st.markdown('<h2 class="subheader">Explanation of Prediction</h2>', unsafe_allow_html=True)
                
                # Create SHAP explainer
                # ----- SHAP Analysis Section -----
                st.markdown('<h2 class="subheader">Explanation of Prediction</h2>', unsafe_allow_html=True)

                # Create SHAP explainer
                with st.spinner("Generating SHAP explanations..."):
                    explainer = shap.TreeExplainer(model)
                    
                    # Convert input data to numpy array
                    input_array = np.array(input_data).reshape(1, -1)
                    
                    # Get feature names
                    feature_names = get_feature_names()
                    
                    # Make sure input_array matches feature_names length
                    if input_array.shape[1] != len(feature_names):
                        st.warning(f"Input dimensions ({input_array.shape[1]}) don't match feature names ({len(feature_names)}). Adjusting...")
                        
                        # If input is larger than feature names, truncate input
                        if input_array.shape[1] > len(feature_names):
                            input_array = input_array[:, :len(feature_names)]
                        # If feature names are more than input, truncate feature names
                        else:
                            feature_names = feature_names[:input_array.shape[1]]
                    
                    # Calculate SHAP values with the possibly adjusted input
                    shap_values = explainer.shap_values(input_array)
                    
                    # Debug information
                    st.write(f"Input shape: {input_array.shape}")
                    st.write(f"Feature names count: {len(feature_names)}")
                    
                    # Handle different SHAP value formats
                    if isinstance(shap_values, list):
                        # For multi-class models
                        class_index = min(int(prediction), len(shap_values)-1)
                        shap_values_for_instance = shap_values[class_index][0]
                        st.write(f"SHAP values shape (class {class_index}): {shap_values[class_index].shape}")
                    else:
                        # For binary classification or regression
                        shap_values_for_instance = shap_values[0]
                        st.write(f"SHAP values shape: {shap_values.shape}")
                    
                    # Ensure we have a flat array
                    shap_values_for_instance = np.array(shap_values_for_instance).flatten()
                    st.write(f"Flattened SHAP values length: {len(shap_values_for_instance)}")
                    
                    # Final dimension check
                    if len(shap_values_for_instance) != len(feature_names):
                        st.warning("SHAP values length doesn't match feature names. Adjusting dimensions...")
                        # Take the shorter length
                        min_len = min(len(shap_values_for_instance), len(feature_names))
                        shap_values_for_instance = shap_values_for_instance[:min_len]
                        feature_names_adjusted = feature_names[:min_len]
                        st.write(f"Adjusted to length: {min_len}")
                    else:
                        feature_names_adjusted = feature_names
                    
                    # Get readable values for explanation
                    readable_values = get_readable_values(input_data)
                    
                    # Display text explanation
                    st.markdown("#### Key Factors Influencing This Prediction")
                    st.markdown(generate_text_explanation(shap_values_for_instance, feature_names_adjusted, readable_values), unsafe_allow_html=True)
                    
                    # SHAP visualization tabs
                    shap_tabs = st.tabs(["Force Plot", "Waterfall Plot", "Feature Importance", "Decision Plot"])
                    
                    with shap_tabs[0]:
                        st.markdown("#### Force Plot")
                        st.markdown("Shows how each feature pushes the prediction from the base value.")
                        
                        # Use adjusted dimensions
                        force_plot = create_shap_force_plot(explainer, shap_values_for_instance, input_array[0, :len(shap_values_for_instance)], feature_names_adjusted)
                        st.pyplot(force_plot)
                        
                    with shap_tabs[1]:
                        st.markdown("#### Waterfall Plot")
                        st.markdown("Visualizes how each feature contributes to push the model output from the base value to the final prediction.")
                        
                        # Use adjusted dimensions
                        waterfall_plot = create_shap_waterfall_plot(explainer, shap_values_for_instance, input_array[0, :len(shap_values_for_instance)], feature_names_adjusted)
                        st.pyplot(waterfall_plot)
                        
                    with shap_tabs[2]:
                        st.markdown("#### Feature Importance")
                        st.markdown("Shows which features are most important for this prediction.")
                        
                        # For the bar plot, use the adjusted dimensions
                        bar_plot = create_shap_bar_plot(explainer, input_array[:, :len(feature_names_adjusted)], feature_names_adjusted)
                        st.pyplot(bar_plot)
                        
                    with shap_tabs[3]:
                        st.markdown("#### Decision Plot")
                        st.markdown("Shows the path from the base value to the final prediction.")
                        
                        # Use adjusted dimensions
                        decision_plot = create_shap_decision_plot(explainer, shap_values_for_instance, input_array[0, :len(shap_values_for_instance)], feature_names_adjusted)
                        st.pyplot(decision_plot)
            else:
                st.error("Failed to load the prediction model. Please try again later.")
    else:
        # Placeholder visual when no prediction has been made
        st.markdown("""
        <div style="text-align: center; padding: 50px 0;">
            <img src="https://img.icons8.com/fluency/96/000000/health-data.png" style="opacity: 0.3;">
            <p style="color: #95a5a6; margin-top: 20px;">Enter patient details and click 'Predict' to see results</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)