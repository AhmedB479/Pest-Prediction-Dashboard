import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random

# === Load Models ===
model_paths = {
    "W_FLY": './models/tuned_models/stacking_W_FLY_model.pkl',
    "JASSID": './models/tuned_models/random_forest_JASSID_model.pkl',
    "THRIPS": './models/tuned_models/stacking_THRIPS_model.pkl',
    "MBUG": './models/tuned_models/knn_M_BUG_model.pkl',
    "MITES": './models/tuned_models/knn_MITES_model.pkl',
    "APHIDS": './models/tuned_models/adaboost_APHIDS_model.pkl',
    "DUSKY": './models/tuned_models/voting_DUSKY_COTTON_BUG_model.pkl',
    "SBW": './models/tuned_models/voting_SBW_model.pkl',
    "PBW": './models/tuned_models/xgboost_PBW_model.pkl',
    "ABW": './models/tuned_models/voting_ABW_model.pkl',
    "ARMYWORM": './models/tuned_models/stacking_Army_Worm_model.pkl',
}
models = {key: joblib.load(path) for key, path in model_paths.items()}

# === Tehsils List ===
TEHSILS = [ 'haroon abad', 'gojra', 'jhang', 'bhera', 'rojhan', 'kot chutta', 'faisalabad',
    'chicha watni', 'kabir wala', 'sumundri', 'sahiwal', 'shujabad', 'bhakkar', 'liaquat pur',
    'ali pur', 'muzaffargarh', 'khanewal', 'piplan', 'rahim yar khan', 'arif wala', 'a.p. sial',
    'bahawalnagar', 'kallor kot', 'vehari', 'd.g. khan', 'mianwali', 'jatoi', 'depalpur',
    'multan', 'jahanian', 'shorkot', 'dunya pur', 'hasilpur', 'chishtian', 'tandlianwala',
    'karor lal esan', 'kehror pacca', 'sadiqabad', 'chobara', 't.t. singh', 'burewala', 'yazman',
    'pak pattan', 'lodhran', 'darya khan', 'taunsa', 'isa khel', 'renala khurd', 'layyah',
    'rajanpur', 'khan pur', 'minchinabad', 'okara', '18-hazari', 'kot addu', 'a.p.east',
    'mailsi', 'jampur', 'bahawalpur', 'mankera', 'mian channu', 'quaidabad', 'jalal pur p.w',
    'bhowana', 'kasur', 'fort abbas', 'kot momin', 'sargodha', 'khushab', 'chak jhumra',
    'pir mahal', 'sillanwali', 'chunian', 'patoki', 'jaranwala', 'bhalwal', 'shahpur',
    'kot radha kishan'
]

BASE_TEMP = 10
MAX_TEMP = 30

def calculate_gdd(temp_mean, temp_max, temp_min):
    return round(max(min(temp_mean, MAX_TEMP) - BASE_TEMP, 0), 2)

def generate_random_inputs():
    return {
        'week': random.randint(1, 4),
        'month': random.randint(1, 12),
        'year': random.randint(2015, 2030),
        'tehsil': random.choice(TEHSILS),
        'spots_visited': random.randint(10, 30),
        'area_visited': round(random.uniform(20, 100), 1),
        'temp_mean': round(random.uniform(25, 35), 1),
        'temp_max': round(random.uniform(30, 40), 1),
        'temp_min': round(random.uniform(20, 28), 1),
        'dew_point': round(random.uniform(15, 25), 1),
        'nitrogen': random.randint(80, 150),
        'phosphorus': random.randint(30, 60),
        'potassium': random.randint(150, 250),
        'humidity': random.randint(50, 85),
        'ph': round(random.uniform(5.5, 7.5), 1),
        'rainfall': random.randint(0, 200)
    }

def predict(input_data):
    return {key: round(model.predict(input_data)[0], 2) for key, model in models.items()}

# === Streamlit UI ===
st.set_page_config(page_title="Cotton Pest Predictor", layout="wide")
st.title("🌱 Cotton Pest Prediction System")

col1, col2 = st.columns(2)
with col1:
    week = st.number_input("Week (1-4)", 1, 4, 1)
    month = st.number_input("Month", 1, 12, 6)
    year = st.number_input("Year", 2000, 2100, 2023)
    tehsil = st.selectbox("Tehsil", TEHSILS, index=TEHSILS.index("faisalabad"))
    spots_visited = st.number_input("Total Spots Visited", 1, 100, 15)
    area_visited = st.number_input("Total Area Visited (acres)", 0.0, 500.0, 50.0)

with col2:
    temp_mean = st.number_input("Mean Temperature (°C)", 0.0, 50.0, 28.5)
    temp_max = st.number_input("Max Temperature (°C)", 0.0, 50.0, 35.0)
    temp_min = st.number_input("Min Temperature (°C)", 0.0, 50.0, 22.0)
    dew_point = st.number_input("Dew Point (°C)", 0.0, 50.0, 18.0)
    nitrogen = st.number_input("Nitrogen (N)", 0, 300, 120)
    phosphorus = st.number_input("Phosphorus (P)", 0, 100, 45)
    potassium = st.number_input("Potassium (K)", 0, 300, 200)
    humidity = st.slider("Humidity (%)", 0, 100, 70)
    ph = st.slider("Soil pH", 0.0, 14.0, 6.5)
    rainfall = st.number_input("Rainfall (mm)", 0, 500, 150)

# Calculate GDDs
daily_gdd = calculate_gdd(temp_mean, temp_max, temp_min)
weekly_gdd = daily_gdd * 7
cumulative_gdd = daily_gdd * (week + (month - 1) * 4)

# Predict Button
if st.button("🔍 Predict Pest Levels"):
    tehsil_encoded = TEHSILS.index(tehsil)
    input_df = pd.DataFrame([[
        week, month, year, tehsil_encoded,
        spots_visited, area_visited,
        temp_mean, temp_max, temp_min, dew_point,
        daily_gdd, weekly_gdd, cumulative_gdd,
        nitrogen, phosphorus, potassium,
        humidity, ph, rainfall
    ]], columns=[
        'week', 'month', 'year', 'TEHSILS',
        'TOTAL SPOTS VISITED', 'TOTAL AREA VISITED',
        'temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min', 'dew_point_2m_mean',
        'daily_gdd', 'weekly_gdd', 'cumulative_gdd',
        'N', 'P', 'K', 'humidity', 'ph', 'rainfall'
    ])

    preds = predict(input_df)

    st.subheader("📈 Pest Level Predictions (%)")
    for k, v in preds.items():
        st.write(f"**{k.replace('_', ' ')}**: {v} %")

    st.subheader("🌡️ GDD Metrics")
    st.write(f"**Daily GDD:** {daily_gdd}")
    st.write(f"**Weekly GDD:** {weekly_gdd}")
    st.write(f"**Cumulative GDD:** {cumulative_gdd}")

# Generate Random Inputs
if st.button("🎲 Generate Random Inputs"):
    r = generate_random_inputs()
    st.experimental_rerun()
