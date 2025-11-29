# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Konfigurasi halaman
st.set_page_config(
    page_title="Bike Sharing Prediction Dashboard",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

class BikeSharingDashboard:
    def __init__(self):
        self.load_models()
        
    def load_models(self):
        """Load trained pipeline models"""
        try:
            self.day_pipeline = joblib.load('rf_day_model.pkl')
            self.hour_pipeline = joblib.load('rf_hour_model.pkl')
            st.success("✅ Models loaded successfully!")
            
            # Debug: Check feature requirements
            st.info(f"DAY model expects: {self.day_pipeline.n_features_in_} features")
            st.info(f"HOUR model expects: {self.hour_pipeline.n_features_in_} features")
            
        except Exception as e:
            st.error(f"❌ Error loading models: {e}")
            st.stop()
    
    def get_expected_features(self, dataset_type):
        """Get the expected feature names for each model"""
        if dataset_type == 'day':
            return ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 
                   'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
        else:  # hour
            return ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 
                   'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
    
    def preprocess_input(self, input_data, dataset_type):
        """Preprocess user input for prediction"""
        # Map categorical features
        season_map = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
        weather_map = {
            1: 'Clear/Partly Cloudy',
            2: 'Misty/Cloudy', 
            3: 'Light Snow/Rain',
            4: 'Heavy Rain/Snow'
        }
        month_map = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        weekday_map = {
            0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday',
            4: 'Thursday', 5: 'Friday', 6: 'Saturday'
        }
        
        # Get expected features for this dataset type
        expected_features = self.get_expected_features(dataset_type)
        
        # Create feature dictionary with ALL expected features
        features = {}
        for feature in expected_features:
            if feature in input_data:
                features[feature] = input_data[feature]
            else:
                st.error(f"Missing feature: {feature}")
                return None, None
        
        # Create DataFrame with correct column order
        df = pd.DataFrame([features])
        df = df[expected_features]  # Ensure correct column order
        
        # Create display info
        display_info = {
            'Season': season_map[input_data['season']],
            'Year': 2011 if input_data['yr'] == 0 else 2012,
            'Month': month_map[input_data['mnth']],
            'Holiday': 'Yes' if input_data['holiday'] == 1 else 'No',
            'Weekday': weekday_map[input_data['weekday']],
            'Working Day': 'Yes' if input_data['workingday'] == 1 else 'No',
            'Weather': weather_map[input_data['weathersit']],
            'Temperature': f"{input_data['temp']:.2f}",
            'Feels Like': f"{input_data['atemp']:.2f}",
            'Humidity': f"{input_data['hum']:.2f}",
            'Windspeed': f"{input_data['windspeed']:.2f}"
        }
        
        if dataset_type == 'hour':
            display_info['Hour'] = f"{input_data['hr']:02d}:00"
        
        return df, display_info
    
    def make_prediction(self, input_df, dataset_type):
        """Make prediction using the appropriate model"""
        if dataset_type == 'day':
            prediction = self.day_pipeline.predict(input_df)
            return prediction[0]
        else:
            prediction = self.hour_pipeline.predict(input_df)
            return prediction[0]
    
    def create_input_sidebar(self):
        """Create input form in sidebar"""
        st.sidebar.header("🚲 Bike Sharing Parameters")
        
        # Dataset type selection
        dataset_type = st.sidebar.radio(
            "Prediction Type",
            ['Daily', 'Hourly'],
            index=0,
            help="Choose between daily or hourly prediction"
        )
        
        # Convert to model type
        model_type = 'day' if dataset_type == 'Daily' else 'hour'
        
        # Time features
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            season = st.selectbox(
                "Season",
                [1, 2, 3, 4],
                index=0,
                format_func=lambda x: ['Spring', 'Summer', 'Fall', 'Winter'][x-1]
            )
            
            year = st.selectbox(
                "Year",
                [0, 1],
                index=0,
                format_func=lambda x: '2011' if x == 0 else '2012'
            )
            
            month = st.slider("Month", 1, 12, 6)
            
        with col2:
            holiday = st.selectbox(
                "Holiday",
                [0, 1],
                index=0,
                format_func=lambda x: 'No' if x == 0 else 'Yes'
            )
            
            weekday = st.selectbox(
                "Weekday",
                [0, 1, 2, 3, 4, 5, 6],
                index=0,
                format_func=lambda x: [
                    'Sunday', 'Monday', 'Tuesday', 'Wednesday',
                    'Thursday', 'Friday', 'Saturday'
                ][x]
            )
            
            workingday = st.selectbox(
                "Working Day",
                [0, 1],
                index=0,
                format_func=lambda x: 'No' if x == 0 else 'Yes'
            )
        
        # Hour input for hourly predictions - INI YANG PERLU DITAMBAHKAN
        hour = 12  # default value
        if dataset_type == 'Hourly':
            hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
        
        # Weather features
        st.sidebar.subheader("🌤️ Weather Conditions")
        
        weather = st.sidebar.selectbox(
            "Weather Situation",
            [1, 2, 3, 4],
            index=0,
            format_func=lambda x: [
                'Clear/Partly Cloudy',
                'Misty/Cloudy', 
                'Light Snow/Rain',
                'Heavy Rain/Snow'
            ][x-1]
        )
        
        temp = st.sidebar.slider(
            "Temperature (normalized)",
            0.0, 1.0, 0.5,
            help="Normalized temperature (0-1 scale)"
        )
        
        atemp = st.sidebar.slider(
            "Feels Like Temperature (normalized)",
            0.0, 1.0, 0.5,
            help="Normalized 'feels like' temperature (0-1 scale)"
        )
        
        humidity = st.sidebar.slider(
            "Humidity (normalized)",
            0.0, 1.0, 0.5,
            help="Normalized humidity (0-1 scale)"
        )
        
        windspeed = st.sidebar.slider(
            "Windspeed (normalized)", 
            0.0, 1.0, 0.1,
            help="Normalized windspeed (0-1 scale)"
        )
        
        # Prepare input data - PASTIKAN SEMUA FEATURE ADA
        input_data = {
            'season': season,
            'yr': year,
            'mnth': month,
            'holiday': holiday,
            'weekday': weekday,
            'workingday': workingday,
            'weathersit': weather,
            'temp': temp,
            'atemp': atemp,
            'hum': humidity,
            'windspeed': windspeed
        }
        
        # TAMBAHKAN hr JIKA HOURLY PREDICTION
        if dataset_type == 'Hourly':
            input_data['hr'] = hour
        
        return input_data, model_type
    
    def plot_prediction_analysis(self, prediction, dataset_type):
        """Create visualization for prediction analysis"""
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["📊 Prediction Analysis", "📈 Trends", "🌡️ Weather Impact"])
        
        with tab1:
            self._plot_prediction_gauge(prediction, dataset_type)
        
        with tab2:
            self._plot_seasonal_trends(dataset_type)
            
        with tab3:
            self._plot_weather_impact(dataset_type)
    
    def _plot_prediction_gauge(self, prediction, dataset_type):
        """Plot gauge chart for prediction"""
        # Set different ranges for day vs hour predictions
        if dataset_type == 'day':
            max_val = 10000
            thresholds = [3000, 6000]
        else:
            max_val = 1000
            thresholds = [300, 600]
            
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = prediction,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Predicted Bike Rentals"},
            gauge = {
                'axis': {'range': [None, max_val]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, thresholds[0]], 'color': "lightgray"},
                    {'range': [thresholds[0], thresholds[1]], 'color': "lightblue"},
                    {'range': [thresholds[1], max_val], 'color': "blue"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': prediction
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_seasonal_trends(self, dataset_type):
        """Plot seasonal trends"""
        # Sample data for demonstration
        seasons = ['Spring', 'Summer', 'Fall', 'Winter']
        if dataset_type == 'day':
            avg_rentals = [2500, 4500, 4200, 3000]
        else:
            avg_rentals = [150, 280, 260, 180]
            
        fig = px.bar(
            x=seasons, 
            y=avg_rentals,
            title="Average Bike Rentals by Season",
            labels={'x': 'Season', 'y': 'Average Rentals'},
            color=avg_rentals,
            color_continuous_scale='Blues'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_weather_impact(self, dataset_type):
        """Plot weather impact on rentals"""
        weather_conditions = ['Clear', 'Misty', 'Light Rain', 'Heavy Rain']
        if dataset_type == 'day':
            rentals = [4500, 3500, 2000, 800]
        else:
            rentals = [280, 220, 120, 50]
            
        fig = px.pie(
            values=rentals,
            names=weather_conditions,
            title="Weather Impact on Rentals",
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_feature_explanations(self):
        """Show feature explanations"""
        with st.expander("📖 Feature Explanations"):
            st.markdown("""
            **Feature Descriptions:**
            - **Season**: 1=Spring, 2=Summer, 3=Fall, 4=Winter
            - **Year**: 0=2011, 1=2012  
            - **Month**: 1=January to 12=December
            - **Hour** (Hourly only): 0-23 (for hourly predictions)
            - **Holiday**: Whether day is holiday (1=Yes, 0=No)
            - **Weekday**: 0=Sunday to 6=Saturday
            - **Working Day**: Neither weekend nor holiday (1=Yes, 0=No)
            - **Weather Situation**: 
                - 1: Clear, Few clouds, Partly cloudy
                - 2: Mist + Cloudy, Mist + Broken clouds
                - 3: Light Snow, Light Rain + Thunderstorm
                - 4: Heavy Rain + Thunderstorm
            - **Temperature**: Normalized temperature in Celsius (0-1)
            - **Feels Like**: Normalized "feels like" temperature (0-1)
            - **Humidity**: Normalized relative humidity (0-1)
            - **Windspeed**: Normalized wind speed (0-1)
            """)
    
    def run(self):
        """Run the dashboard"""
        # Header
        st.markdown('<h1 class="main-header">🚲 Bike Sharing Prediction Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar for inputs
        input_data, dataset_type = self.create_input_sidebar()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎯 Prediction Results")
            
            # Make prediction when button is clicked
            if st.button("🔮 Predict Bike Rentals", type="primary", use_container_width=True):
                with st.spinner("Making prediction..."):
                    # Preprocess input
                    input_df, display_info = self.preprocess_input(input_data, dataset_type)
                        
                    if input_df is not None:
                        # Debug info
                        st.write(f"📊 Using {dataset_type} model")
                        st.write(f"🔢 Input features: {len(input_df.columns)}")
                        st.write(f"📋 Features: {list(input_df.columns)}")
                        
                        # Make prediction
                        prediction = self.make_prediction(input_df, dataset_type)
                        
                        # Display results
                        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                        
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.metric(
                                label="Predicted Bike Rentals", 
                                value=f"{prediction:.0f}",
                                help=f"Predicted number of bikes to be rented"
                            )
                        
                        with col_b:
                            dataset_label = "per day" if dataset_type == 'day' else "per hour"
                            st.metric(
                                label="Prediction Type",
                                value=dataset_label.title(),
                            )
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Show input parameters
                        st.subheader("📋 Input Parameters")
                        param_cols = st.columns(3)
                        params_list = list(display_info.items())
                        
                        for i, (key, value) in enumerate(params_list):
                            with param_cols[i % 3]:
                                st.info(f"**{key}:** {value}")
                        
                        # Visualization
                        st.subheader("📈 Analysis & Insights")
                        self.plot_prediction_analysis(prediction, dataset_type)
                    else:
                        st.error("❌ Error in preparing input data. Please check all features are provided.")
        
        with col2:
            st.subheader("ℹ️ Dashboard Info")
            st.markdown("""
            <div class="metric-card">
            <h4>About This Dashboard</h4>
            <p>Predict bike rental demand using machine learning models trained on historical data.</p>
            <p><strong>Models:</strong> Random Forest</p>
            <p><strong>Data:</strong> Bike Sharing Dataset</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Current prediction type info
            current_type = "Daily" if dataset_type == 'day' else "Hourly"
            st.markdown(f"""
            <div class="metric-card">
            <h4>Current Settings</h4>
            <p><strong>Prediction Type:</strong> {current_type}</p>
            <p><strong>Features Required:</strong> {len(self.get_expected_features(dataset_type))}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature explanations
            self.show_feature_explanations()
            
            # Sample predictions
            st.subheader("💡 Sample Scenarios")
            
            if st.button("🏖️ Summer Weekend", use_container_width=True):
                st.session_state.season = 2  # Summer
                st.session_state.weekday = 0  # Sunday
                st.session_state.holiday = 0
                st.session_state.workingday = 0
                st.session_state.weathersit = 1  # Clear
                st.session_state.temp = 0.8
                st.session_state.atemp = 0.75
                st.rerun()
            
            if st.button("🌧️ Rainy Workday", use_container_width=True):
                st.session_state.season = 1  # Spring  
                st.session_state.weekday = 1  # Monday
                st.session_state.holiday = 0
                st.session_state.workingday = 1
                st.session_state.weathersit = 3  # Light Rain
                st.session_state.temp = 0.3
                st.session_state.atemp = 0.25
                st.rerun()

            if st.button("❄️ Winter Evening", use_container_width=True):
                st.session_state.season = 4  # Winter
                st.session_state.weekday = 4  # Thursday
                st.session_state.holiday = 0
                st.session_state.workingday = 1
                st.session_state.weathersit = 2  # Misty
                st.session_state.temp = 0.1
                st.session_state.atemp = 0.05
                if dataset_type == 'hour':
                    st.session_state.hr = 18  # Evening
                st.rerun()

# Run the dashboard
if __name__ == "__main__":
    # Initialize session state for form persistence
    default_values = {
        'season': 1,
        'yr': 0,
        'mnth': 6,
        'hr': 12,
        'holiday': 0,
        'weekday': 0,
        'workingday': 0,
        'weathersit': 1,
        'temp': 0.5,
        'atemp': 0.5,
        'hum': 0.5,
        'windspeed': 0.1
    }
    
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    dashboard = BikeSharingDashboard()
    dashboard.run()