
import requests
from datetime import datetime, timedelta
import streamlit as st
from PIL import Image
import joblib
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import keras
from tensorflow.keras.models import load_model
import branca.colormap
from functools import lru_cache


# Set page configuration
st.set_page_config(
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
        /* General page styling */
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f8f9fa;
        }
        .section {
            padding: 50px 5%;
        }
        .hero {
            background-size: cover;
            color: white;
            text-align: center;
            padding: 100px 5%;
        }
        .hero h1 {
            font-size: 3rem;
            font-weight: bold;
        }
        .hero p {
            font-size: 1.5rem;
            margin: 20px 0;
        }
        .card {
            background-color:#f3fcf3 ;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin: 20px;
            text-align: center;
        }
        .card h3 {
            color: #333333;
            font-size: 1.8rem;
        }
        .card p {
            font-size: 1.2rem;
            color: #555555;
        }
        .contact-section {
            background-color: #343a40;
            color: white;
            padding: 70px 5%;
            border-radius: 10px;
        }
        .contact-section a {
            color: #17a2b8;
            text-decoration: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load model and scaler
@st.cache_resource
def load_trained_model():
    return load_model("app/assets/flood_model.keras")

@st.cache_resource
def load_scaler():
    return joblib.load("app/assets/scaler.pkl")

# Load model and scaler once
model = load_trained_model()
scaler = load_scaler()


def validate_coordinates(lat, lng):
    """Ensure valid geographic coordinates with proper error messages"""
    try:
        if not (-90 <= float(lat) <= 90):
            st.error(f"Invalid latitude: {lat}. Must be between -90 and 90")
            return False
        if not (-180 <= float(lng) <= 180):
            st.error(f"Invalid longitude: {lng}. Must be between -180 and 180")
            return False
        return True
    except ValueError:
        st.error("Coordinates must be numeric values")
        return False

district_coordinates = {
    "Bagerhat": {"X_COR": 22.651568, "Y_COR": 89.785938},
    "Bandarban": {"X_COR": 22.195327, "Y_COR": 92.218377},
    "Barguna": {"X_COR": 22.156889, "Y_COR": 90.329871},
    "Barisal": {"X_COR": 22.701002, "Y_COR": 90.353451},
    "Bhola": {"X_COR": 22.687946, "Y_COR": 90.644397},
    "Bogra": {"X_COR": 24.846522, "Y_COR": 89.377755},
    "Brahmanbaria": {"X_COR": 23.957090, "Y_COR": 91.111928},
    "Chandpur": {"X_COR": 23.233258, "Y_COR": 90.671291},
    "Chittagong": {"X_COR": 22.356851, "Y_COR": 91.783182},
    "Chuadanga": {"X_COR": 23.640196, "Y_COR": 88.841841},
    "Comilla": {"X_COR": 23.460856, "Y_COR": 91.180909},
    "Cox's Bazar": {"X_COR": 21.427229, "Y_COR": 92.005806},
    "Dhaka": {"X_COR": 23.810331, "Y_COR": 90.412521},
    "Dinajpur": {"X_COR": 25.627858, "Y_COR": 88.633576},
    "Faridpur": {"X_COR": 23.607082, "Y_COR": 89.842940},
    "Feni": {"X_COR": 23.015915, "Y_COR": 91.397600},
    "Gaibandha": {"X_COR": 25.328751, "Y_COR": 89.528088},
    "Gazipur": {"X_COR": 23.999940, "Y_COR": 90.420273},
    "Gopalganj": {"X_COR": 23.005085, "Y_COR": 89.826605},
    "Habiganj": {"X_COR": 24.374945, "Y_COR": 91.415530},
    "Jamalpur": {"X_COR": 24.937218, "Y_COR": 89.937774},
    "Jessore": {"X_COR": 23.166667, "Y_COR": 89.208611},
    "Jhalokathi": {"X_COR": 22.640562, "Y_COR": 90.198739},
    "Jhenaidah": {"X_COR": 23.544817, "Y_COR": 89.153921},
    "Joypurhat": {"X_COR": 25.102347, "Y_COR": 89.021263},
    "Khagrachari": {"X_COR": 23.119285, "Y_COR": 91.984663},
    "Khulna": {"X_COR": 22.845641, "Y_COR": 89.540328},
    "Kishoreganj": {"X_COR": 24.444937, "Y_COR": 90.776575},
    "Kurigram": {"X_COR": 25.805445, "Y_COR": 89.636174},
    "Kushtia": {"X_COR": 23.901258, "Y_COR": 89.120482},
    "Lakshmipur": {"X_COR": 22.942477, "Y_COR": 90.841184},
    "Lalmonirhat": {"X_COR": 25.992346, "Y_COR": 89.284725},
    "Madaripur": {"X_COR": 23.164102, "Y_COR": 90.189680},
    "Magura": {"X_COR": 23.487337, "Y_COR": 89.419956},
    "Manikganj": {"X_COR": 23.861733, "Y_COR": 90.004683},
    "Meherpur": {"X_COR": 23.762213, "Y_COR": 88.631821},
    "Moulvibazar": {"X_COR": 24.482934, "Y_COR": 91.777417},
    "Munshiganj": {"X_COR": 23.542217, "Y_COR": 90.530500},
    "Mymensingh": {"X_COR": 24.747149, "Y_COR": 90.420273},
    "Naogaon": {"X_COR": 24.913159, "Y_COR": 88.753095},
    "Narail": {"X_COR": 23.172534, "Y_COR": 89.512672},
    "Narayanganj": {"X_COR": 23.623810, "Y_COR": 90.499844},
    "Narsingdi": {"X_COR": 23.932233, "Y_COR": 90.715421},
    "Natore": {"X_COR": 24.420556, "Y_COR": 89.000282},
    "Netrokona": {"X_COR": 24.870955, "Y_COR": 90.727887},
    "Nilphamari": {"X_COR": 25.931794, "Y_COR": 88.856006},
    "Noakhali": {"X_COR": 22.869563, "Y_COR": 91.099398},
    "Pabna": {"X_COR": 23.998542, "Y_COR": 89.233646},
    "Panchagarh": {"X_COR": 26.341100, "Y_COR": 88.554160},
    "Patuakhali": {"X_COR": 22.359631, "Y_COR": 90.329871},
    "Pirojpur": {"X_COR": 22.584126, "Y_COR": 89.972030},
    "Rajbari": {"X_COR": 23.757430, "Y_COR": 89.644466},
    "Rajshahi": {"X_COR": 24.374945, "Y_COR": 88.604255},
    "Rangamati": {"X_COR": 22.732374, "Y_COR": 92.198329},
    "Rangpur": {"X_COR": 25.743892, "Y_COR": 89.275227},
    "Satkhira": {"X_COR": 22.7185, "Y_COR": 89.0705},
    "Chapai Nawabganj": {"X_COR": 24.6833, "Y_COR": 88.2500},
    "Sherpur": {"X_COR": 25.0200, "Y_COR": 90.0170},
    "Shariatpur": {"X_COR": 23.2423, "Y_COR": 90.4348}
}
 


def search_now_page():
    st.markdown("""
    <style>
        .prediction-header {
            background: linear-gradient(145deg, #56ccf2, #2f80ed);
            color: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }
        .input-card {
            background: #fff;
            border-radius: 15px;
            padding: 25px;
            margin: 15px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .stNumberInput>div>div>input {
            border-radius: 15px;
            padding: 12px 15px;
            border: 2px solid #27ae60;
        }
        .prediction-button {
            background: linear-gradient(45deg, #3498db, #2980b9);!important;
            color: white !important;
            border-radius: 25px !important;
            padding: 15px 35px !important;
            font-size: 1.1rem !important;
            transition: all 0.3s !important;
        }
        .prediction-button:hover {
            transform: scale(1.05) !important;
            box-shadow: 0 5px 15px rgba(52,152,219,0.4) !important;
        }
        .result-card {
           background: linear-gradient(145deg, #a8c0ff, #3f6be5);  /* Light blue gradient */
border-radius: 15px;
padding: 25px;
margin: 25px 0;
border-left: 5px solid #f39c12;  /* Lighter border color for contrast */
color: #2c3e50;  /* Dark text color for contrast */

        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="prediction-header"><h1>🌦️ Flood Risk Prediction Analysis</h1></div>', unsafe_allow_html=True)
    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            district = st.selectbox("Select District", list(district_coordinates.keys()))

            max_temp = st.number_input("Max Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
            min_temp = st.number_input("Min Temperature (°C)", min_value=0.0, max_value=50.0, value=15.0)
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=50.0)
            relative_humidity = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
            wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=10.0)

        with col2:
            cloud_coverage = st.number_input("Cloud Coverage (%)", min_value=0.0, max_value=100.0, value=30.0)
            bright_sunshine = st.number_input("Bright Sunshine (hours)", min_value=0.0, max_value=24.0, value=6.0)
            alt = st.number_input("Altitude (m)", min_value=0.0, max_value=5000.0, value=100.0)

            x_cor = district_coordinates[district]["X_COR"]
            y_cor = district_coordinates[district]["Y_COR"]

            st.write(f"**📍 Coordinates for {district}:**")
            st.write(f"X: {x_cor}, Y: {y_cor}")

        month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=6)     

    if st.button("🌧️ Predict Flood Risk", key="predict_button"):
        columns_to_scale = [
            'Max Temp', 'Min Temp', 'Rainfall', 'Relative Humidity',
            'Wind Speed', 'Cloud Coverage', 'Bright Sunshine', 
            'X_COR', 'Y_COR', 'ALT'
        ]
        input_data = pd.DataFrame([[
            max_temp, min_temp, rainfall, relative_humidity,
            wind_speed, cloud_coverage, bright_sunshine,
            x_cor, y_cor, alt
        ]], columns=columns_to_scale)
        
        scaled_features = scaler.transform(input_data)
        
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        processed_features = np.concatenate([
            scaled_features,
            np.array([[month_sin, month_cos]])
        ], axis=1)
        
        input_array = processed_features.reshape(1, processed_features.shape[1], 1)
        
        probability = model.predict(input_array)[0][0]
        risk_level = "High Risk" if probability >= 0.5 else "Low Risk"
        
        st.markdown(f'''
            <div class="result-card">
                <h3>Prediction Result</h3>
                <p style="font-size:1.4rem; color: {'#c0392b' if risk_level == 'High Risk' else '#27ae60'}">
                    {risk_level} ({probability:.2%} probability)
                </p>
                <small>Always stay prepared and follow local authority guidelines!!</small>
            </div>
        ''', unsafe_allow_html=True)

    # if st.button("🔄 Fetch Current Weather Conditions"):
    #     try:
    #         district_coords = district_coordinates[district]
    #         lat = district_coords["Y_COR"]
    #         lng = district_coords["X_COR"]
            
    #         if not validate_coordinates(lat, lng):
    #             return
                
    #         # Use cached API call with validation
    #         live_data = cached_flood_data(lat, lng)

    #         if live_data and 'data' in live_data and len(live_data['data']) > 0:
    #             # Get latest entry
    #             latest = live_data['data'][0]
                
    #             # Update session state with safe defaults
    #             st.session_state.rainfall = latest.get('precipitation', 0)
    #             st.session_state.wind_speed = latest.get('windSpeed', 0)
    #             st.session_state.rel_humidity = latest.get('humidity', 60)
    #             st.rerun()
    #         else:
    #             st.warning("No live data available for this region")
                
    #     except Exception as e:
    #         st.error(f"Failed to fetch weather data: {str(e)}")



from streamlit_folium import st_folium  

def flood_prone_areas_page():
    st.title("Flood-Prone Areas")
    st.write("Explore the regions in Bangladesh that are most vulnerable to flooding.")
    
    df = pd.read_csv("app/assets/flood_dataset.csv")
    
    new_df = df.drop(['A', 'Period', 'LATITUDE', 'LONGITUDE', 'Station Number'], axis=1)
    new_df.rename(columns={'Station Names': 'District'}, inplace=True)
    Train_df = new_df[~new_df['District'].isin(['Ishurdi', 'Maijdee Court'])].copy()
    original_coords = Train_df[['X_COR', 'Y_COR']].copy()  
    Train_df.drop(['District', 'YEAR'], axis=1, inplace=True)

    scaler = joblib.load("app/assets/scaler.pkl")
    columns_to_scale = ['Max Temp', 'Min Temp', 'Rainfall', 'Relative Humidity',
                        'Wind Speed', 'Cloud Coverage', 'Bright Sunshine', 'X_COR', 'Y_COR', 'ALT']
    Train_df[columns_to_scale] = scaler.transform(Train_df[columns_to_scale])
    
    Train_df['month_sin'] = np.sin(2 * np.pi * Train_df['Month'] / 12)
    Train_df['month_cos'] = np.cos(2 * np.pi * Train_df['Month'] / 12)
    Train_df.drop('Month', axis=1, inplace=True)
    
    features = np.array(Train_df)
    features_reshaped = features.reshape(features.shape[0], features.shape[1], 1)
    
    model = load_model("app/assets/flood_model.keras")
    predictions = model.predict(features_reshaped).flatten() 

    results_df = original_coords.copy()
    results_df['Flood_Probability'] = predictions
    
    m = folium.Map(location=[23.6850, 90.3563], zoom_start=7)
    marker_cluster = MarkerCluster().add_to(m)
    

    for idx, row in results_df.iterrows():
        folium.CircleMarker(
            location=[row['Y_COR'], row['X_COR']], 
            radius=5,
            color='red' if row['Flood_Probability'] >= 0.5 else 'blue', 
            fill=True,
            fill_opacity=0.7,
            popup=f"Flood Risk: {row['Flood_Probability']:.2f}"
        ).add_to(marker_cluster)
    
    st_folium(m, width=1000, height=500)
 

def notifications_page():
   
    st.markdown("""
    <style>
        .notifications-header {
            background: linear-gradient(145deg, #2c3e50, #3498db);
            color: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }
        .warning-card {
            background: linear-gradient(145deg, #a2d9ce, #5dade2);
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 4px solid #e74c3c;
            transition: transform 0.2s;
            color: #b71c1c;  /* Deep red color */
            font-weight: 600;
            text-shadow: 1px 1px 2px rgba(255,255,255,0.3);
        }
        .warning-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .warning-card::before {
            content: "⚠️";
            margin-right: 10px;
            font-size: 1.2em;
        }
        .subscribe-form {
            background: linear-gradient(145deg, #2d3436, #3a6073);
            padding: 25px;
            border-radius: 15px;
            margin: 25px 0;
        }
        .stTextInput>div>div>input {
            border-radius: 25px;
            padding: 12px 20px;
            border: 2px solid #3498db;
            color: #2c3e50;  /* Dark text color */
        }
        .stButton>button {
            background: linear-gradient(45deg, #3498db, #2980b9);
            color: white;
            border-radius: 25px;
            padding: 12px 30px;
            border: none;
            font-weight: bold;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(52,152,219,0.4);
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="notifications-header"><h1>🌊 Flood Alert Notifications</h1></div>', unsafe_allow_html=True)
    
    with st.container():
        # st.markdown('<div class="subscribe-form">', unsafe_allow_html=True)
        email = st.text_input("Enter your email to receive alerts:", key="email_input")
        location=st.text_input("Enter your location:",key="location_input")
        col1, col2 = st.columns([2,1])
        with col1:
            if st.button("🔔 Subscribe for Alerts"):
                st.success(f"Thank you for subscribing! Alerts will be sent to {email}.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("📢 Recent Flood Warnings")
    warnings = [
        "Flood warning issued for Dhaka on 2023-10-15.",
        "Moderate flood risk in Sylhet on 2023-10-14.",
        "No active warnings for Chittagong.",
    ]
    for warning in warnings:
        st.markdown(f'<div class="warning-card">{warning}</div>', unsafe_allow_html=True)
    
#    # Modified alert section
#     st.subheader(f"Recent Flood Warnings ({current_time})")
    
#     # Get updated data if available
#     if not st.session_state.results_df.empty:
#         high_risk = st.session_state.results_df[st.session_state.results_df['Flood_Probability'] > 0.7]
#         if not high_risk.empty:
#             for _, row in high_risk.iterrows():
#                 st.markdown(f'''
#                 <div class="warning-card">
#                     {row['District']} - {row['Flood_Probability']:.2%} flood risk
#                 </div>
#                 ''', unsafe_allow_html=True)
#         else:
#             st.success("No active flood warnings currently")
#     else:
#         st.info("Flood data not loaded yet. Visit Flood-Prone Areas page first.")

    
    
    
# Home Page
def home_page():
    st.markdown(
        '''
        <div style="text-align: center; background: linear-gradient(135deg, #004d1a, #014d40 , #006400 ); padding: 150px 20px; border-radius: 10px; position: relative; box-shadow: 0 6px 25px rgba(0, 0, 0, 0.4);">
            <div style="display: flex; justify-content: center; align-items: center; font-family: 'Georgia', serif; color: #D0F0C0; position: relative;text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.9);">
                <span style="font-size: 40px; font-weight: bold; margin-left: 20px; font-style: italic; font-family: 'Georgia'; letter-spacing: 2px;">FLOODGUARD</span>
            </div>
            <h1 style="font-family: 'Georgia', serif; color: white; font-size: 32px; margin-top: 50px; text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.5);">
                Precision-Driven Flood Forecasting for Bangladesh
            </h1>
            <div style="position: absolute; top: 40px; right: 20px; color: white; display: flex; justify-content: space-around;">
                <a href="?page=search" style="margin: 0 15px; color: white; text-decoration: none; font-weight: bold; font-size: 16px; transition: color 0.3s ease;">Explore Forecasts</a>
                <a href="?page=flood" style="margin: 0 15px; color: white; text-decoration: none; font-weight: bold; font-size: 16px; transition: color 0.3s ease;">Vulnerable Regions</a>
                <a href="?page=notifications" style="margin: 0 15px; color: white; text-decoration: none; font-weight: bold; font-size: 16px; transition: color 0.3s ease;">Real-Time Alerts</a>
            </div>
            <h6 style="font-family: 'Georgia', serif; color: #C0C0C0; font-size: 20px; margin-top: 10px; text-shadow: 1px 1px 5px rgba(0, 0, 0, 0.3);">
                Empowering communities with timely, accurate, and actionable flood insights.
            </h6>
            <button style="background-color: #ffffff; color: #013220; font-size: 16px; padding: 12px 24px; border-radius: 30px; border: none; cursor: pointer; margin-top: 20px; font-weight: bold; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2); transition: background-color 0.3s ease;">
                Stay Informed – Register Now
            </button>
        </div>
        ''',
        unsafe_allow_html=True,
    )
    st.write("")  # Empty line for spacing
    st.write("")
    st.markdown(
        '''
        <div class="main" style="padding: 50px; background-color: #F0FFF0; border-radius: 10px; margin-top: 30px;">
            <h2 style="font-style: italic; font-family: 'Georgia', 'Times New Roman', serif; font-size: 20px; color: #2c3e50; text-align: center; margin: 20px;">
            Using Machine Learning to Make 
            Critical Flood Forecasting Information
            Universally Accessible in Bangladesh!
            </h2>
            <p style="text-align: center; color: #555;">Floods claim thousands of lives annually. Beyond the immediate loss of life, floods displace over 20 million people worldwide every year, according to the United Nations, and cause financial damages exceeding $100 billion annually. In Bangladesh, one of the most flood-prone countries globally, nearly 80 percent of the landmass is classified as floodplain, making large portions of the population vulnerable to devastating impacts, including the destruction of homes, crops, and infrastructure.</p>
            <p style="text-align: center; color: #555;">We leverage Machine Learning (ML) and Deep Learning (DL) to combat these challenges by enabling real-time flood forecasting and risk assessment. By integrating satellite imagery, hydrological data, and advanced prediction models, our solution aims to improve early warning systems, reduce response times, and empower communities with actionable insights to minimize casualties and economic losses.</p>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    st.write("")  
    st.write("")
    # Display flood image
    image1_path = "app/assets/flood_image.jpeg"  # Update this path
    image2_path = "app/assets/11.jpg"  # Update this path
    try:
        header_image1 = Image.open(image1_path)
        header_image2 = Image.open(image2_path)
        col1, col2 = st.columns([1,1])  # Adjust column ratios as needed
        with col1:
            st.image(header_image1, caption='Aerial View of a Flooded Region', use_container_width=True)
        with col2:
            st.image(header_image2, caption='Flooded Agricultural Land Devastating Crops',use_container_width=True)
    except FileNotFoundError:
        st.error(f"Header image not found at {header_image_path}")

    # Add a Feature Highlights Section
    st.markdown(
        '''
        <div style="padding: 50px; background-color: #f9f9f9; border-radius: 10px; margin-top: 30px;">
            <h2 style="text-align: center; font-family: 'Georgia', serif; color: #2c3e50;">Why Choose FloodGuard?</h2>
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap; margin-top: 30px;">
                <!-- Card 1 -->
                <div style="width: 30%; background: white; padding: 20px; margin: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-radius: 10px;">
                    <h3 style="text-align: center; color: #004d40; font-family: 'Arial'; margin-top: 15px;">Real-Time Forecasting</h3>
                    <p style="text-align: center; color: #555;">Stay updated with accurate, real-time flood forecasting to prepare and respond effectively.</p>
                </div>
                <!-- Card 2 -->
                <div style="width: 30%; background: white; padding: 20px; margin: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-radius: 10px;">
                    <h3 style="text-align: center; color: #004d40; font-family: 'Arial'; margin-top: 15px;">Machine Learning Insights</h3>
                    <p style="text-align: center; color: #555;">Harness the power of ML for accurate flood predictions using historical and environmental data.</p>
                </div>
                <!-- Card 3 -->
                <div style="width: 30%; background: white; padding: 20px; margin: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-radius: 10px;">
                    <h3 style="text-align: center; color: #004d40; font-family: 'Arial'; margin-top: 15px;">Community Support</h3>
                    <p style="text-align: center; color: #555;">Empower local communities with actionable insights to minimize flood impacts and losses.</p>
                </div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )
    st.write("")
    st.write("")

    # Flood Probability Section
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.header("Flood Probabilities by Region")
    st.write("Stay aware of the flood probabilities across various regions in Bangladesh.")

    # Display flood probabilities in cards
    flood_data = {
        "Dhaka": "12.21%",
        "Chittagong": "7.08%",
        "Sylhet": "9.15%",
        "Barisal": "8.2%",
        "Cumilla": "6.7%",
        "Rangpur": "13.93%",
        "Dinajpur": "5.15%",
    }

    col1, col2, col3 = st.columns(3)
    for i, (region, probability) in enumerate(flood_data.items()):
        with [col1, col2, col3][i % 3]:
            st.markdown(
                f"""
                <div class="card">
                    <h3>{region}</h3>
                    <p>{probability} Probability</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)

    # Register Button
    if st.button("Login Now For Your Area!"):
        st.success("Thank you for registering!")

    # Add a "How It Works" section
    st.markdown(
        '''
        <div style="padding: 50px; background-color: #ffffff; border-radius: 0px; margin-top: 30px;">
            <h2 style="text-align: center; font-family: 'Georgia', serif; color: #2c3e50;">How Does It Work?</h2>
            <p style="font-size: 16px; color: #555; text-align: justify; margin: 20px auto; max-width: 800px;">
                The Hydrologic Model identifies whether a river is expected to flood by processing publicly available data sources, such as precipitation and weather data, and outputs a forecast for the water level in the river for the next few days. 
                The Inundation Model simulates how water will spread across the floodplain based on the hydrology forecast and satellite imagery. This enables us to predict the specific areas at risk and the expected water levels.
            </p>
            <h3 style="font-family: 'Georgia', serif; color: #004d40; margin-top: 40px; text-align: center;">Key Features</h3>
            <ul style="list-style-type: disc; color: #555; margin: 20px auto; max-width: 800px; font-size: 16px;">
                <li>Accurate flood predictions up to 7 days in advance using advanced hydrological modeling.</li>
                <li>Actionable forecasts to empower governments, relief organizations, and communities.</li>
                <li>Coverage for data-scarce locations, leveraging satellite and global weather products.</li>
            </ul>
            <div style="padding: 20px; background-color: #f1f8e9; border-radius: 10px; margin-top: 20px; text-align: center;">
                <h4 style="color: #004d40;">Tip: Stay Prepared!</h4>
                <p style="font-size: 14px; color: #555;">
                    Stay updated with flood forecasts for your region by accessing our platform or signing up for notifications. Share these insights with your community to minimize risks.
                </p>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )
    st.image("app/assets/how_it_works.jpeg", caption="Flood Prediction Workflow", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    # FAQs section
    st.subheader("FAQs")
    faqs = {
        "What is the purpose of this platform?": "To provide accurate flood detection and prediction for various locations across the country, helping communities stay prepared and safe.",
        "How does the flood detection system work?": "Our platform processes weather, precipitation, and satellite data to predict water levels and assess flood risks in real-time.",
        "Can this system be used in remote areas?": "Yes! The platform leverages satellite imagery and global weather products, making it effective even in data-scarce regions.",
        "How often are the flood forecasts updated?": "Flood forecasts are updated daily to ensure timely and accurate information.",
        "Can I receive flood alerts for my area?": "Yes! You can sign up for notifications to stay informed about flood risks in your region.",
    }
    for question, answer in faqs.items():
        with st.expander(question):
            st.write(answer)
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    # Contact Section
    st.markdown(
        """
        <div class="contact-section">
            <h2>Contact Us</h2>
            <p><strong>Phone:</strong> +880 123 456 7890</p>
            <p><strong>Email:</strong> <a href="mailto:info@floodforecast.com">info@floodforecast.com</a></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Footer Section
    st.markdown(
        """
        <div class="footer">
            © 2025 Flood Prediction Bangladesh. All rights reserved.
        </div>
        """,
        unsafe_allow_html=True,
    )

# Main function to handle navigation
def main():
    st.sidebar.title("Navigation")
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "Home"
    
    # Check URL parameters
    query_params = st.query_params
    if "page" in query_params:
        page_mapping = {
            "search": "Search Now",
            "flood": "Flood-Prone Areas",
            "notifications": "Notifications"
        }
        st.session_state.page = page_mapping.get(query_params["page"], "Home")
    
    # Create radio buttons that sync with session state
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Search Now", "Flood-Prone Areas", "Notifications"],
        index=["Home", "Search Now", "Flood-Prone Areas", "Notifications"].index(st.session_state.page)
    )
    
    # Update session state if radio changes
    if page != st.session_state.page:
        st.session_state.page = page
        st.query_params.clear()
        st.rerun()
    
    # Page routing
    if st.session_state.page == "Home":
        home_page()
    elif st.session_state.page == "Search Now":
        search_now_page()
    elif st.session_state.page == "Flood-Prone Areas":
        flood_prone_areas_page()
    elif st.session_state.page == "Notifications":
        notifications_page()

if __name__ == "__main__":
    main()
# import datetime
# import streamlit as st
# from PIL import Image
# import joblib
# import pandas as pd
# import numpy as np
# import folium
# from streamlit_folium import folium_static
# from streamlit_folium import folium_static
# from tensorflow.keras.models import load_model  # Added import for Keras load_model
# import branca.colormap

# # Set page configuration
# st.set_page_config(
#     page_icon="🌊",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # Custom CSS for styling
# st.markdown(
#     """
#     <style>
#         /* General page styling */
#         body {
#             font-family: 'Arial', sans-serif;
#             background-color: #f8f9fa;
#         }
#         .section {
#             padding: 50px 5%;
#         }
#         .hero {
#             background-size: cover;
#             color: white;
#             text-align: center;
#             padding: 100px 5%;
#         }
#         .hero h1 {
#             font-size: 3rem;
#             font-weight: bold;
#         }
#         .hero p {
#             font-size: 1.5rem;
#             margin: 20px 0;
#         }
#         .card {
#             background-color:#f3fcf3 ;
#             border-radius: 10px;
#             box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#             padding: 20px;
#             margin: 20px;
#             text-align: center;
#         }
#         .card h3 {
#             color: #333333;
#             font-size: 1.8rem;
#         }
#         .card p {
#             font-size: 1.2rem;
#             color: #555555;
#         }
#         .contact-section {
#             background-color: #343a40;
#             color: white;
#             padding: 70px 5%;
#             border-radius: 10px;
#         }
#         .contact-section a {
#             color: #17a2b8;
#             text-decoration: none;
#         }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # Load your trained model
# # @st.cache_resource
# # def load_model():
# #     return joblib.load("app/assets/flood_model.pkl")  # Update with your model path

# # model = load_model()

# # # Function to make predictions
# # def predict_flood(input_data):
# #     prediction = model.predict(input_data)
# #     return prediction

# # Load model and scaler
# @st.cache_resource
# def load_trained_model():
#     return load_model("app/assets/flood_model.keras")

# @st.cache_resource
# def load_scaler():
#     return joblib.load("app/assets/scaler.pkl")

# # Load model and scaler once
# model = load_trained_model()
# scaler = load_scaler()

# # # Search Now Page
# # def search_now_page():
# #     st.title("Search Now")
# #     st.write("Enter the required details to get flood predictions for your area.")

# #     # Example input fields
# #     rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=50.0)
# #     river_level = st.number_input("River Water Level (m)", min_value=0.0, max_value=20.0, value=5.0)
# #     temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)

# #     if st.button("Predict Flood Risk"):
# #         input_data = np.array([[rainfall, river_level, temperature]])
# #         prediction = predict_flood(input_data)
# #         st.success(f"Flood Risk Prediction: {prediction[0]}")


# def search_now_page():
#     st.title("Flood Risk Prediction")
#     st.write("Enter the required details to get flood predictions for your area.")

#     # Input fields
#     max_temp = st.number_input("Max Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
#     min_temp = st.number_input("Min Temperature (°C)", min_value=0.0, max_value=50.0, value=15.0)
#     rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=50.0)
#     relative_humidity = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
#     wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=10.0)
#     cloud_coverage = st.number_input("Cloud Coverage (%)", min_value=0.0, max_value=100.0, value=30.0)
#     bright_sunshine = st.number_input("Bright Sunshine (hours)", min_value=0.0, max_value=24.0, value=6.0)
#     alt = st.number_input("Altitude (m)", min_value=0.0, max_value=5000.0, value=100.0)
#     month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=6)

#     if st.button("Predict Flood Risk"):
#         # Calculate cyclic features from the month
#         month_sin = np.sin(2 * np.pi * month / 12)
#         month_cos = np.cos(2 * np.pi * month / 12)
        
#         # Prepare input for the scaler (8 features only)
#         raw_input = np.array([[max_temp, min_temp, rainfall, relative_humidity,
#                               wind_speed, cloud_coverage, bright_sunshine, alt]])
#         # Scale the original 8 features
#         scaled_input = scaler.transform(raw_input)
        
#         # Append the cyclic features (month_sin and month_cos)
#         final_input = np.append(scaled_input, [[month_sin, month_cos]], axis=1)
        
#         # Predict using the full 10 features
#         prediction = model.predict(final_input)
#         risk_level = "High Risk" if prediction[0][0] > 0.5 else "Low Risk"
#         st.success(f"Flood Risk Prediction: {risk_level} ({prediction[0][0]:.2f} probability)")

# def flood_prone_areas_page():
#     st.title("Flood-Prone Areas")
#     st.write("Explore the regions in Bangladesh that are most vulnerable to flooding.")

#     # Create a map centered on Bangladesh
#     m = folium.Map(location=[23.6850, 90.3563], zoom_start=7)

#     # Add markers for flood-prone areas
#     flood_prone_areas = {
#         "Dhaka": (23.8103, 90.4125),
#         "Chittagong": (22.3569, 91.7832),
#         "Sylhet": (24.8949, 91.8687),
#         "Barisal": (22.7010, 90.3535),
#     }

#     for area, coords in flood_prone_areas.items():
#         folium.Marker(coords, popup=area).add_to(m)

#     # Display the map
#     folium_static(m)
# # def flood_prone_areas_page():
# #     st.title("Flood-Prone Areas")
# #     st.write("Explore flood risk predictions across Bangladesh based on current conditions")

# #     # Load district data with average features
# #     @st.cache_data
# #     def load_district_data():
# #         try:
# #             df = pd.read_csv("app/assets/district_features.csv")
# #             required_columns = ['District', 'LATITUDE', 'LONGITUDE', 'Max Temp', 'Min Temp',
# #                                'Rainfall', 'Relative Humidity', 'Wind Speed', 'Cloud Coverage',
# #                                'Bright Sunshine', 'ALT']
# #             if not all(col in df.columns for col in required_columns):
# #                 missing = [col for col in required_columns if col not in df.columns]
# #                 raise ValueError(f"Missing columns: {missing}")
# #             return df
# #         except Exception as e:
# #             st.error(f"Error loading data: {str(e)}")
# #             return pd.DataFrame()

# #     district_data = load_district_data()
    
# #     if district_data.empty:
# #         st.info("Please ensure district_features.csv is properly formatted")
# #         return

# #     # User inputs
# #     col1, col2 = st.columns(2)
# #     with col1:
# #         month = st.slider("Select Month", 1, 12, value=datetime.now().month)
# #     with col2:
# #         risk_threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.5, 0.05)

# #     # Calculate cyclic features
# #     month_sin = np.sin(2 * np.pi * month / 12)
# #     month_cos = np.cos(2 * np.pi * month / 12)

# #     # Prepare features for prediction
# #     feature_columns = ['Max Temp', 'Min Temp', 'Rainfall', 'Relative Humidity',
# #                       'Wind Speed', 'Cloud Coverage', 'Bright Sunshine', 'ALT']
    
# #     try:
# #         # Scale features
# #         raw_features = district_data[feature_columns]
# #         scaled_features = scaler.transform(raw_features)
        
# #         # Add temporal features
# #         scaled_features = np.hstack([
# #             scaled_features,
# #             np.full((len(scaled_features), 1), month_sin),
# #             np.full((len(scaled_features), 1), month_cos)
# #         ])
        
# #         # Make predictions
# #         predictions = model.predict(scaled_features)
# #         district_data['Flood_Probability'] = predictions
# #         district_data['Risk_Level'] = np.where(predictions >= risk_threshold, 'High', 'Low')

# #         # Create map
# #         m = folium.Map(location=[23.6850, 90.3563], zoom_start=6.4, tiles='CartoDB positron')
# #         colormap = branca.colormap.LinearColormap(
# #             colors=['green', 'yellow', 'red'],
# #             vmin=0,
# #             vmax=1,
# #             caption='Flood Probability'
# #         )

# #         # Add markers
# #         for _, row in district_data.iterrows():
# #             prob = row['Flood_Probability']
# #             folium.CircleMarker(
# #                 location=[row['LATITUDE'], row['LONGITUDE']],
# #                 radius=8 + prob * 25,  # Dynamic size based on probability
# #                 color=colormap(prob),
# #                 fill=True,
# #                 fill_opacity=0.7,
# #                 popup=folium.Popup(
# #                     f"<strong>{row['District']}</strong><br>"
# #                     f"Predicted Risk: {prob:.2%}<br>"
# #                     f"Current Threshold: {risk_threshold:.0%}",
# #                     max_width=250
# #                 ),
# #                 tooltip=f"{row['District']} - {prob:.2%}"
# #             ).add_to(m)

# #         # Add legend
# #         colormap.add_to(m)
        
# #         # Display map
# #         folium_static(m, width=1200, height=600)

# #         # Risk interpretation guide
# #         st.markdown(f"""
# #         ### Map Interpretation (Threshold: {risk_threshold:.0%})
# #         - 🔴 High Risk Areas: {len(district_data[district_data['Risk_Level'] == 'High'])} districts
# #         - 🟢 Low Risk Areas: {len(district_data[district_data['Risk_Level'] == 'Low'])} districts
# #         - **Circle Size**: Larger circles indicate higher predicted flood probability
# #         - **Color Intensity**: Shows relative risk level from green (low) to red (high)
# #         """)

# #     except Exception as e:
# #         st.error(f"Prediction error: {str(e)}")
# # def flood_prone_areas_page():
# #     st.title("Flood-Prone Areas")
# #     st.write("Explore the regions in Bangladesh that are most vulnerable, based on our model predictions.")

# #     import pandas as pd
# #     import numpy as np
# #     import folium
# #     from streamlit_folium import folium_static

# #     # Load the dataset
# #     try:
# #         df = pd.read_csv("app/assets/flood dataset.csv")
# #     except Exception as e:
# #         st.error("Could not load 'flood dataset.csv'. Please ensure the file is in the correct location.")
# #         return

# #     # Clean up column names (remove extra spaces)
# #     df.columns = [col.strip() for col in df.columns]

# #     # If you have latitude and longitude columns, save them for later mapping
# #     if "LATITUDE" not in df.columns or "LONGITUDE" not in df.columns:
# #         st.error("The dataset does not have 'LATITUDE' and 'LONGITUDE' columns needed for mapping.")
# #         return

# #     # Save coordinates for mapping and then drop them from features (if needed for prediction)
# #     coords = df[['LATITUDE', 'LONGITUDE']]

# #     # Drop unnecessary columns (modify as per your data; note: do NOT drop LATITUDE and LONGITUDE if you want to map)
# #     # For example, if you need them for mapping later, keep them aside:
# #     df_features = df.drop(['A', 'Period', 'X_COR', 'Y_COR', 'Station Number'], axis=1, errors='ignore')

# #     # If there is a 'Month' column, apply cyclic encoding
# #     if "Month" in df_features.columns:
# #         df_features['month_sin'] = np.sin(2 * np.pi * df_features['Month'] / 12)
# #         df_features['month_cos'] = np.cos(2 * np.pi * df_features['Month'] / 12)
# #         df_features = df_features.drop(columns=['Month'])
# #     else:
# #         st.error("The 'Month' column is missing in the dataset.")
# #         return

# #     # Scale numeric features using the loaded scaler
# #     columns_to_scale = ['Max Temp', 'Min Temp', 'Rainfall', 'Relative Humidity',
# #                           'Wind Speed', 'Cloud Coverage', 'Bright Sunshine', 'ALT']
# #     try:
# #         df_features[columns_to_scale] = scaler.transform(df_features[columns_to_scale])
# #     except Exception as e:
# #         st.error("Error during scaling. Check that the 'scaler.pkl' file is available and the columns match.")
# #         return

# #     # Remove non-numeric columns from features (if any) that are not needed for prediction
# #     # For this example, we assume LATITUDE and LONGITUDE are not used in prediction
# #     if "LATITUDE" in df_features.columns:
# #         df_features = df_features.drop(columns=['LATITUDE'])
# #     if "LONGITUDE" in df_features.columns:
# #         df_features = df_features.drop(columns=['LONGITUDE'])

# #     # Prepare the features for prediction
# #     X = df_features.values
# #     X_input = X.reshape(X.shape[0], X.shape[1], 1)

# #     # Make predictions using your pre-loaded model
# #     try:
# #         predictions = model.predict(X_input)
# #     except Exception as e:
# #         st.error("Error during model prediction. Please verify that the model was loaded correctly.")
# #         return

# #     # Create a new column for predicted risk (1 if probability > 0.5, otherwise 0)
# #     pred_flags = (predictions.flatten() > 0.5).astype(int)
# #     df['predicted_risk'] = pred_flags

# #     # Extract rows flagged as vulnerable
# #     vulnerable_rows = df[df['predicted_risk'] == 1]

# #     st.write("Number of vulnerable entries as predicted by our model:", len(vulnerable_rows))

# #     # Create a Folium map centered on Bangladesh
# #     m = folium.Map(location=[23.6850, 90.3563], zoom_start=7)

# #     # Add markers for each vulnerable row using the saved coordinates
# #     for idx, row in vulnerable_rows.iterrows():
# #         lat = row["LATITUDE"]
# #         lon = row["LONGITUDE"]
# #         # Optionally, include additional info (e.g., a station name or a computed value)
# #         folium.Marker(location=[lat, lon], popup=f"Risk: {row['predicted_risk']}").add_to(m)

# #     # Render the map in Streamlit
# #     folium_static(m)

# # Notifications Page
# def notifications_page():
#     st.title("Notifications")
#     st.write("Sign up to receive flood alerts or view recent warnings.")

#     # Example: Sign up for alerts
#     email = st.text_input("Enter your email to receive alerts:")
#     if st.button("Subscribe"):
#         st.success(f"Thank you for subscribing! Alerts will be sent to {email}.")

#     # Example: Display recent warnings
#     st.subheader("Recent Flood Warnings")
#     warnings = [
#         "Flood warning issued for Dhaka on 2023-10-15.",
#         "Moderate flood risk in Sylhet on 2023-10-14.",
#         "No active warnings for Chittagong.",
#     ]
#     for warning in warnings:
#         st.write(warning)

# # Home Page
# def home_page():
#     st.markdown(
#         '''
#         <div style="text-align: center; background: linear-gradient(135deg, #004d1a, #014d40 , #006400 ); padding: 150px 20px; border-radius: 10px; position: relative; box-shadow: 0 6px 25px rgba(0, 0, 0, 0.4);">
#             <div style="display: flex; justify-content: center; align-items: center; font-family: 'Georgia', serif; color: #D0F0C0; position: relative;text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.9);">
#                 <span style="font-size: 40px; font-weight: bold; margin-left: 20px; font-style: italic; font-family: 'Georgia'; letter-spacing: 2px;">FLOODGUARD</span>
#             </div>
#             <h1 style="font-family: 'Georgia', serif; color: white; font-size: 32px; margin-top: 50px; text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.5);">
#                 Precision-Driven Flood Forecasting for Bangladesh
#             </h1>
#             <div style="position: absolute; top: 40px; right: 20px; color: white; display: flex; justify-content: space-around;">
#                 <a href="#search" style="margin: 0 15px; color: white; text-decoration: none; font-weight: bold; font-size: 16px; transition: color 0.3s ease;">Explore Forecasts</a>
#                 <a href="#flood-prone-areas" style="margin: 0 15px; color: white; text-decoration: none; font-weight: bold; font-size: 16px; transition: color 0.3s ease;">Vulnerable Regions</a>
#                 <a href="#notifications" style="margin: 0 15px; color: white; text-decoration: none; font-weight: bold; font-size: 16px; transition: color 0.3s ease;">Real-Time Alerts</a>
#             </div>
#             <h6 style="font-family: 'Georgia', serif; color: #C0C0C0; font-size: 20px; margin-top: 10px; text-shadow: 1px 1px 5px rgba(0, 0, 0, 0.3);">
#                 Empowering communities with timely, accurate, and actionable flood insights.
#             </h6>
#             <button style="background-color: #ffffff; color: #013220; font-size: 16px; padding: 12px 24px; border-radius: 30px; border: none; cursor: pointer; margin-top: 20px; font-weight: bold; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2); transition: background-color 0.3s ease;">
#                 Stay Informed – Register Now
#             </button>
#         </div>
#         ''',
#         unsafe_allow_html=True,
#     )


#     st.write("")  # Empty line for spacing
#     st.write("")

#     # # Display header image
#     # header_image_path = "app/assets/logo.jpeg"  # Update this path
#     # try:
#     #     header_image = Image.open(header_image_path)
#     #     col1, col2, col3 = st.columns([1, 2, 1])  # Adjust column ratios as needed
#     #     with col1:
#     #         st.write("")  # Empty column for spacing
#     #     with col2:
#     #         st.image(header_image, caption='Flood Forecasting', use_container_width=True)
#     #     with col3:
#     #         st.write("")  # Empty column for spacing
#     # except FileNotFoundError:
#     #     st.error(f"Header image not found at {header_image_path}")



#     st.markdown(
#         '''
#         <div class="main" style="padding: 50px; background-color: #F0FFF0; border-radius: 10px; margin-top: 30px;">
#             <h2 style="font-style: italic; font-family: 'Georgia', 'Times New Roman', serif; font-size: 20px; color: #2c3e50; text-align: center; margin: 20px;">
#             Using Machine Learning to Make 
#             Critical Flood Forecasting Information
#             Universally Accessible in Bangladesh!
#             </h2>
#             <p style="text-align: center; color: #555;">Floods claim thousands of lives annually. Beyond the immediate loss of life, floods displace over 20 million people worldwide every year, according to the United Nations, and cause financial damages exceeding $100 billion annually. In Bangladesh, one of the most flood-prone countries globally, nearly 80 percent of the landmass is classified as floodplain, making large portions of the population vulnerable to devastating impacts, including the destruction of homes, crops, and infrastructure.</p>
#             <p style="text-align: center; color: #555;">We leverage Machine Learning (ML) and Deep Learning (DL) to combat these challenges by enabling real-time flood forecasting and risk assessment. By integrating satellite imagery, hydrological data, and advanced prediction models, our solution aims to improve early warning systems, reduce response times, and empower communities with actionable insights to minimize casualties and economic losses.</p>
#         </div>
#         ''',
#         unsafe_allow_html=True,
#     )


#     st.write("")  
#     st.write("")

#     # Display flood image
#     image1_path = "app/assets/flood_image.jpeg"  # Update this path
#     image2_path = "app/assets/11.jpg"  # Update this path

#     try:
#         header_image1 = Image.open(image1_path)
#         header_image2 = Image.open(image2_path)
#         col1, col2 = st.columns([1,1])  # Adjust column ratios as needed
#         with col1:
#             st.image(header_image1, caption='Aerial View of a Flooded Region', use_container_width=True)
#         with col2:
#             st.image(header_image2, caption='Flooded Agricultural Land Devastating Crops',use_container_width=True)
#     except FileNotFoundError:
#         st.error(f"Header image not found at {header_image_path}")

#     # Add a Feature Highlights Section
#     st.markdown(
#         '''
#         <div style="padding: 50px; background-color: #f9f9f9; border-radius: 10px; margin-top: 30px;">
#             <h2 style="text-align: center; font-family: 'Georgia', serif; color: #2c3e50;">Why Choose FloodGuard?</h2>
#             <div style="display: flex; justify-content: space-around; flex-wrap: wrap; margin-top: 30px;">
#                 <!-- Card 1 -->
#                 <div style="width: 30%; background: white; padding: 20px; margin: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-radius: 10px;">
#                     <h3 style="text-align: center; color: #004d40; font-family: 'Arial'; margin-top: 15px;">Real-Time Forecasting</h3>
#                     <p style="text-align: center; color: #555;">Stay updated with accurate, real-time flood forecasting to prepare and respond effectively.</p>
#                 </div>
#                 <!-- Card 2 -->
#                 <div style="width: 30%; background: white; padding: 20px; margin: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-radius: 10px;">
#                     <h3 style="text-align: center; color: #004d40; font-family: 'Arial'; margin-top: 15px;">Machine Learning Insights</h3>
#                     <p style="text-align: center; color: #555;">Harness the power of ML for accurate flood predictions using historical and environmental data.</p>
#                 </div>
#                 <!-- Card 3 -->
#                 <div style="width: 30%; background: white; padding: 20px; margin: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-radius: 10px;">
#                     <h3 style="text-align: center; color: #004d40; font-family: 'Arial'; margin-top: 15px;">Community Support</h3>
#                     <p style="text-align: center; color: #555;">Empower local communities with actionable insights to minimize flood impacts and losses.</p>
#                 </div>
#             </div>
#         </div>
#         ''',
#         unsafe_allow_html=True,
#     )
#     st.write("")
#     st.write("")

#     # Flood Probability Section
#     st.markdown('<div class="section">', unsafe_allow_html=True)
#     st.header("Flood Probabilities by Region")
#     st.write("Stay aware of the flood probabilities across various regions in Bangladesh.")

#     # Display flood probabilities in cards
#     flood_data = {
#         "Dhaka": "12.21%",
#         "Chittagong": "7.08%",
#         "Sylhet": "9.15%",
#         "Barisal": "8.2%",
#         "Cumilla": "6.7%",
#         "Rangpur": "13.93%",
#         "Dinajpur": "5.15%",
#     }

#     col1, col2, col3 = st.columns(3)
#     for i, (region, probability) in enumerate(flood_data.items()):
#         with [col1, col2, col3][i % 3]:
#             st.markdown(
#                 f"""
#                 <div class="card">
#                     <h3>{region}</h3>
#                     <p>{probability} Probability</p>
#                 </div>
#                 """,
#                 unsafe_allow_html=True,
#             )
#     st.markdown('</div>', unsafe_allow_html=True)

#     # Register Button
#     if st.button("Login Now For Your Area!"):
#         st.success("Thank you for registering!")

#     # Add a "How It Works" section
#     st.markdown(
#         '''
#         <div style="padding: 50px; background-color: #ffffff; border-radius: 0px; margin-top: 30px;">
#             <h2 style="text-align: center; font-family: 'Georgia', serif; color: #2c3e50;">How Does It Work?</h2>
#             <p style="font-size: 16px; color: #555; text-align: justify; margin: 20px auto; max-width: 800px;">
#                 The Hydrologic Model identifies whether a river is expected to flood by processing publicly available data sources, such as precipitation and weather data, and outputs a forecast for the water level in the river for the next few days. 
#                 The Inundation Model simulates how water will spread across the floodplain based on the hydrology forecast and satellite imagery. This enables us to predict the specific areas at risk and the expected water levels.
#             </p>
#             <h3 style="font-family: 'Georgia', serif; color: #004d40; margin-top: 40px; text-align: center;">Key Features</h3>
#             <ul style="list-style-type: disc; color: #555; margin: 20px auto; max-width: 800px; font-size: 16px;">
#                 <li>Accurate flood predictions up to 7 days in advance using advanced hydrological modeling.</li>
#                 <li>Actionable forecasts to empower governments, relief organizations, and communities.</li>
#                 <li>Coverage for data-scarce locations, leveraging satellite and global weather products.</li>
#             </ul>
#             <div style="padding: 20px; background-color: #f1f8e9; border-radius: 10px; margin-top: 20px; text-align: center;">
#                 <h4 style="color: #004d40;">Tip: Stay Prepared!</h4>
#                 <p style="font-size: 14px; color: #555;">
#                     Stay updated with flood forecasts for your region by accessing our platform or signing up for notifications. Share these insights with your community to minimize risks.
#                 </p>
#             </div>
#         </div>
#         ''',
#         unsafe_allow_html=True,
#     )

#     st.image("app/assets/how_it_works.jpeg", caption="Flood Prediction Workflow", use_container_width=True)
#     st.markdown('</div>', unsafe_allow_html=True)

#     # FAQs section
#     st.subheader("FAQs")
#     faqs = {
#         "What is the purpose of this platform?": "To provide accurate flood detection and prediction for various locations across the country, helping communities stay prepared and safe.",
#         "How does the flood detection system work?": "Our platform processes weather, precipitation, and satellite data to predict water levels and assess flood risks in real-time.",
#         "Can this system be used in remote areas?": "Yes! The platform leverages satellite imagery and global weather products, making it effective even in data-scarce regions.",
#         "How often are the flood forecasts updated?": "Flood forecasts are updated daily to ensure timely and accurate information.",
#         "Can I receive flood alerts for my area?": "Yes! You can sign up for notifications to stay informed about flood risks in your region.",
#     }
#     for question, answer in faqs.items():
#         with st.expander(question):
#             st.write(answer)

#     st.write("")
#     st.write("")
#     st.write("")
#     st.write("")

#     # Contact Section
#     st.markdown(
#         """
#         <div class="contact-section">
#             <h2>Contact Us</h2>
#             <p><strong>Phone:</strong> +880 123 456 7890</p>
#             <p><strong>Email:</strong> <a href="mailto:info@floodforecast.com">info@floodforecast.com</a></p>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )

#     # Footer Section
#     st.markdown(
#         """
#         <div class="footer">
#             © 2025 Flood Prediction Bangladesh. All rights reserved.
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )

# # Main function to handle navigation
# def main():
#     st.sidebar.title("Navigation")
#     page = st.sidebar.radio("Go to", ["Home", "Search Now", "Flood-Prone Areas", "Notifications"])

#     if page == "Home":
#         home_page()
#     elif page == "Search Now":
#         search_now_page()
#     elif page == "Flood-Prone Areas":
#         flood_prone_areas_page()
#     elif page == "Notifications":
#         notifications_page()

# if __name__ == "__main__":
#     main()