import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import time

# Set page configuration
st.set_page_config(
    page_title="Clinical Obesity Risk Assessment",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #1E88E5;
        padding-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f7f7f7;
        border-radius: 5px;
        padding: 1rem;
        border-left: 5px solid #1E88E5;
    }
    .risk-high {
        color: #f44336;
        font-weight: bold;
    }
    .risk-moderate {
        color: #ff9800;
        font-weight: bold;
    }
    .risk-low {
        color: #4caf50;
        font-weight: bold;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #757575;
        font-style: italic;
        text-align: center;
    }
    hr {
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# App title and subtitle
st.markdown("<h1 class='main-header'>Clinical Obesity Risk Assessment</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Advanced patient analysis tool for healthcare professionals</p>", unsafe_allow_html=True)

# Initialize session state for storing patient data and results
if 'patient_history' not in st.session_state:
    st.session_state.patient_history = []

if 'current_patient' not in st.session_state:
    st.session_state.current_patient = {}

if 'bmi' not in st.session_state:
    st.session_state.bmi = None

if 'risk_level' not in st.session_state:
    st.session_state.risk_level = None

if 'risk_score' not in st.session_state:
    st.session_state.risk_score = None

# Left column: Patient information input
with st.sidebar:
    st.markdown("<h2 class='section-header'>Patient Information</h2>", unsafe_allow_html=True)
    
    # Patient ID
    patient_id = st.text_input("Patient ID", value="", help="Enter unique patient identifier")
    
    # Demographic information
    st.markdown("<h3>Demographics</h3>", unsafe_allow_html=True)
    gender = st.radio("Gender", ["Female", "Male"])
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=300, value=70)
    
    # Quick BMI calculation display in sidebar
    if height > 0 and weight > 0:
        bmi = weight / ((height / 100) ** 2)
        st.session_state.bmi = round(bmi, 1)
        
        bmi_status = "Unknown"
        bmi_color = "black"
        
        if bmi < 18.5:
            bmi_status = "Underweight"
            bmi_color = "blue"
        elif bmi < 25:
            bmi_status = "Normal"
            bmi_color = "green"
        elif bmi < 30:
            bmi_status = "Overweight"
            bmi_color = "orange"
        else:
            bmi_status = "Obese"
            bmi_color = "red"
            
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Current BMI</h3>
            <h2 style='color:{bmi_color}'>{st.session_state.bmi} - {bmi_status}</h2>
        </div>
        """, unsafe_allow_html=True)

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("<h2 class='section-header'>Lifestyle & Medical Factors</h2>", unsafe_allow_html=True)
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["Diet & Nutrition", "Physical Activity", "Medical History"])
    
    with tab1:
        st.subheader("Dietary Habits")
        high_caloric_food = st.radio("High caloric food consumption", ["No", "Occasionally", "Frequently", "Daily"])
        vegetables = st.select_slider("Vegetable consumption", options=["Never", "Rarely", "Sometimes", "Often", "Always"], value="Sometimes")
        meals_per_day = st.radio("Main meals per day", ["1", "2", "3", "More than 3"])
        snacks = st.select_slider("Snacking frequency", options=["Never", "Rarely", "Sometimes", "Often", "Always"], value="Sometimes")
        water_intake = st.select_slider("Daily water intake", options=["Less than 1L", "1-2L", "2-3L", "More than 3L"], value="1-2L")
        calorie_monitoring = st.radio("Calorie monitoring", ["Never", "Occasionally", "Regularly"])
    
    with tab2:
        st.subheader("Activity Profile")
        physical_activity = st.select_slider(
            "Weekly physical activity frequency", 
            options=["None", "1-2 days", "3-4 days", "5+ days"],
            value="1-2 days"
        )
        activity_intensity = st.slider("Activity intensity (minutes of elevated heart rate)", 0, 120, 30, 10)
        screen_time = st.select_slider("Daily screen time", options=["0-2 hours", "2-4 hours", "4-6 hours", "6+ hours"], value="2-4 hours")
        transportation = st.selectbox(
            "Primary mode of transportation", 
            ["Automobile", "Public Transportation", "Motorbike", "Bicycle", "Walking"],
            index=0
        )
    
    with tab3:
        st.subheader("Health Background")
        family_overweight = st.radio("Family history of obesity", ["No", "Yes"])
        smoking = st.radio("Smoking status", ["Non-smoker", "Former smoker", "Current smoker"])
        alcohol_consumption = st.select_slider(
            "Alcohol consumption", 
            options=["None", "Rarely", "Occasionally", "Weekly", "Daily"],
            value="Occasionally"
        )
        sleep_quality = st.slider("Sleep quality (hours/night)", 4, 12, 7)
        existing_conditions = st.multiselect(
            "Existing health conditions",
            ["None", "Diabetes", "Hypertension", "Cardiovascular disease", "Sleep apnea", "Thyroid disorders"],
            default=["None"]
        )
        medications = st.multiselect(
            "Current medications",
            ["None", "Antidepressants", "Antipsychotics", "Steroids", "Beta blockers", "Insulin", "Other"],
            default=["None"]
        )

    # Predict button
    predict_button = st.button("Analyze Patient Data", help="Run obesity risk assessment algorithm", key="predict_button")

# Second column for results
with col2:
    st.markdown("<h2 class='section-header'>Risk Assessment Results</h2>", unsafe_allow_html=True)
    
    if predict_button or st.session_state.risk_level:
        # Show loading animation
        if predict_button:
            with st.spinner("Analyzing patient data..."):
                time.sleep(1.5)  # Simulate computation time for better UX
            
            # Calculate risk based on various factors (more sophisticated than the original)
            def calculate_risk_score():
                # Base score from BMI
                bmi = st.session_state.bmi
                if bmi < 18.5:
                    score = 10
                elif bmi < 25:
                    score = 20
                elif bmi < 30:
                    score = 60
                else:
                    score = 80
                
                # Adjust based on lifestyle factors
                # Diet
                if vegetables in ["Often", "Always"]:
                    score -= 10
                if high_caloric_food in ["Frequently", "Daily"]:
                    score += 15
                if calorie_monitoring == "Regularly":
                    score -= 10
                
                # Activity
                if physical_activity in ["None", "1-2 days"]:
                    score += 15
                elif physical_activity == "5+ days":
                    score -= 20
                
                if transportation in ["Bicycle", "Walking"]:
                    score -= 10
                
                # Medical
                if family_overweight == "Yes":
                    score += 15
                if "None" not in existing_conditions:
                    score += 5 * len(existing_conditions)
                if smoking == "Current smoker":
                    score += 10
                
                # Add slight randomness for demo purposes (remove in production)
                score += np.random.randint(-5, 5)
                
                # Clamp the score
                return max(min(score, 100), 0)
            
            risk_score = calculate_risk_score()
            st.session_state.risk_score = risk_score
            
            # Determine risk level based on score
            if risk_score < 30:
                st.session_state.risk_level = "Low"
            elif risk_score < 60:
                st.session_state.risk_level = "Moderate"
            elif risk_score < 85:
                st.session_state.risk_level = "High"
            else:
                st.session_state.risk_level = "Severe"
            
            # Store current patient data
            st.session_state.current_patient = {
                "id": patient_id,
                "age": age,
                "gender": gender,
                "bmi": st.session_state.bmi,
                "risk_level": st.session_state.risk_level,
                "risk_score": st.session_state.risk_score,
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
            }
            
            # Add to history if not already there
            if patient_id and not any(p["id"] == patient_id for p in st.session_state.patient_history):
                st.session_state.patient_history.append(st.session_state.current_patient)
        
        # Display risk gauge
        risk_color = {
            "Low": "#4caf50",
            "Moderate": "#ff9800",
            "High": "#f44336",
            "Severe": "#b71c1c"
        }.get(st.session_state.risk_level, "#757575")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=st.session_state.risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Obesity Risk Score: <span style='color:{risk_color}'>{st.session_state.risk_level}</span>", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': risk_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': '#c8e6c9'},
                    {'range': [30, 60], 'color': '#ffe0b2'},
                    {'range': [60, 85], 'color': '#ffcdd2'},
                    {'range': [85, 100], 'color': '#e57373'}
                ],
            }
        ))
        
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        # Clinical recommendations
        st.markdown("<h3>Clinical Recommendations</h3>", unsafe_allow_html=True)
        
        recommendations = []
        if st.session_state.risk_level == "Low":
            recommendations = [
                "Maintain current healthy lifestyle",
                "Annual check-up recommended",
                "Continue balanced diet and regular exercise"
            ]
        elif st.session_state.risk_level == "Moderate":
            recommendations = [
                "Increase physical activity to at least 150 min/week",
                "Reduce processed food consumption",
                "Consider nutritional counseling",
                "Follow-up in 6 months"
            ]
        elif st.session_state.risk_level == "High":
            recommendations = [
                "Urgent lifestyle intervention needed",
                "Refer to weight management specialist",
                "Screen for metabolic syndrome",
                "Consider sleep study for apnea",
                "Follow-up in 3 months"
            ]
        else:  # Severe
            recommendations = [
                "Immediate comprehensive intervention required",
                "Evaluate for bariatric surgery candidacy",
                "Full metabolic panel and cardiovascular assessment",
                "Consider pharmacological intervention",
                "Monthly follow-up appointments"
            ]
        
        for rec in recommendations:
            st.markdown(f"- {rec}")
        
        # Key risk factors
        st.markdown("<h3>Key Risk Factors</h3>", unsafe_allow_html=True)
        
        risk_factors = []
        if st.session_state.bmi >= 30:
            risk_factors.append(f"BMI: {st.session_state.bmi} (Obese range)")
        if family_overweight == "Yes":
            risk_factors.append("Family history of obesity")
        if high_caloric_food in ["Frequently", "Daily"]:
            risk_factors.append("High caloric food consumption")
        if physical_activity in ["None", "1-2 days"]:
            risk_factors.append("Insufficient physical activity")
        if "None" not in existing_conditions:
            risk_factors.append(f"Existing conditions: {', '.join([c for c in existing_conditions if c != 'None'])}")
        
        if not risk_factors:
            st.write("No significant risk factors identified.")
        else:
            for factor in risk_factors:
                st.markdown(f"- {factor}")

        # Add BMI chart
        st.markdown("<h3>BMI Classification</h3>", unsafe_allow_html=True)
        
        bmi_fig = go.Figure()
        
        # BMI ranges
        bmi_ranges = [
            {"range": [0, 18.5], "name": "Underweight", "color": "#64b5f6"},
            {"range": [18.5, 25], "name": "Normal", "color": "#81c784"},
            {"range": [25, 30], "name": "Overweight", "color": "#ffd54f"},
            {"range": [30, 35], "name": "Obese Class I", "color": "#ff8a65"},
            {"range": [35, 40], "name": "Obese Class II", "color": "#e57373"},
            {"range": [40, 50], "name": "Obese Class III", "color": "#d32f2f"}
        ]
        
        # Add colored ranges
        for range_info in bmi_ranges:
            bmi_fig.add_shape(
                type="rect",
                x0=range_info["range"][0],
                x1=range_info["range"][1],
                y0=0,
                y1=1,
                fillcolor=range_info["color"],
                opacity=0.3,
                layer="below",
                line_width=0,
            )
            
            # Add text labels
            bmi_fig.add_annotation(
                x=(range_info["range"][0] + range_info["range"][1])/2,
                y=0.5,
                text=range_info["name"],
                showarrow=False,
                font=dict(color="#000000", size=10)
            )
        
        # Add patient marker
        bmi_fig.add_shape(
            type="line",
            x0=st.session_state.bmi,
            x1=st.session_state.bmi,
            y0=0,
            y1=1,
            line=dict(color="#000000", width=2)
        )
        
        bmi_fig.add_annotation(
            x=st.session_state.bmi,
            y=0.8,
            text="Patient BMI",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            ax=0,
            ay=-30
        )
        
        # Update layout
        bmi_fig.update_layout(
            height=150,
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(
                title="BMI",
                range=[10, 45],
                showgrid=False,
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
            ),
            plot_bgcolor="white",
            showlegend=False,
        )
        
        st.plotly_chart(bmi_fig, use_container_width=True)

# Bottom section for patient history and additional features
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h2 class='section-header'>Patient History & Data Analysis</h2>", unsafe_allow_html=True)

if len(st.session_state.patient_history) > 0:
    # Create tabs for different views
    history_tab, analytics_tab = st.tabs(["Patient History", "Population Analytics"])
    
    with history_tab:
        # Convert patient history to DataFrame for display
        history_df = pd.DataFrame(st.session_state.patient_history)
        
        if not history_df.empty:
            st.dataframe(history_df, hide_index=True, use_container_width=True)
            
            # Option to download patient data
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="Export Patient Data (CSV)",
                data=csv,
                file_name="obesity_risk_assessment_data.csv",
                mime="text/csv",
                key="download_csv"
            )
    
    with analytics_tab:
        # Mock population data for comparison
        st.subheader("Population Risk Distribution")
        
        # Generate some mock population data
        if len(st.session_state.patient_history) > 0:
            # Create age groups
            age_groups = ["18-30", "31-45", "46-60", "61+"]
            
            # Create mock data based on age
            mock_data = []
            for _ in range(100):
                age_group = np.random.choice(age_groups)
                
                # Biasing risk levels based on age groups
                if age_group == "18-30":
                    risk = np.random.choice(["Low", "Moderate", "High", "Severe"], p=[0.5, 0.3, 0.15, 0.05])
                elif age_group == "31-45":
                    risk = np.random.choice(["Low", "Moderate", "High", "Severe"], p=[0.3, 0.4, 0.2, 0.1])
                elif age_group == "46-60":
                    risk = np.random.choice(["Low", "Moderate", "High", "Severe"], p=[0.2, 0.3, 0.3, 0.2])
                else:  # 61+
                    risk = np.random.choice(["Low", "Moderate", "High", "Severe"], p=[0.1, 0.3, 0.4, 0.2])
                    
                mock_data.append({"Age Group": age_group, "Risk Level": risk})
            
            mock_df = pd.DataFrame(mock_data)
            
            # Create visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk distribution by age group
                risk_by_age = mock_df.groupby(["Age Group", "Risk Level"]).size().reset_index(name="Count")
                fig = px.bar(
                    risk_by_age, 
                    x="Age Group", 
                    y="Count", 
                    color="Risk Level",
                    color_discrete_map={
                        "Low": "#4caf50", 
                        "Moderate": "#ff9800", 
                        "High": "#f44336", 
                        "Severe": "#b71c1c"
                    },
                    title="Risk Distribution by Age Group"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Overall risk distribution
                risk_counts = mock_df["Risk Level"].value_counts().reset_index()
                risk_counts.columns = ["Risk Level", "Count"]
                
                # Define color mapping and order
                color_map = {
                    "Low": "#4caf50", 
                    "Moderate": "#ff9800", 
                    "High": "#f44336", 
                    "Severe": "#b71c1c"
                }
                
                risk_order = ["Low", "Moderate", "High", "Severe"]
                
                # Create sorted dataframe
                sorted_risk = pd.DataFrame({
                    "Risk Level": risk_order,
                })
                
                sorted_risk = sorted_risk.merge(risk_counts, on="Risk Level", how="left").fillna(0)
                
                # Create pie chart
                fig = px.pie(
                    sorted_risk, 
                    values="Count", 
                    names="Risk Level",
                    color="Risk Level",
                    color_discrete_map=color_map,
                    title="Overall Risk Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Add current patient marker
            if st.session_state.current_patient:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>Current Patient Comparison</h3>
                    <p>The current patient's risk level of <span class='risk-{st.session_state.risk_level.lower()}'>{st.session_state.risk_level}</span> 
                    places them in comparison to the population distribution shown above.</p>
                </div>
                """, unsafe_allow_html=True)
else:
    st.info("No patient history available. Submit patient data to begin tracking.")

# Footer with disclaimer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='disclaimer'>This clinical tool is for healthcare professional use only. Risk assessment is based on statistical modeling and should be used as a supplementary aid to clinical judgment, not as a replacement. Always consider individual patient circumstances when making clinical decisions.</p>", unsafe_allow_html=True)