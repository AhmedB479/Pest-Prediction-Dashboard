import streamlit as st
import pandas as pd
import numpy as np
import random
import os

# === Page Configuration ===
st.set_page_config(
    page_title="Cotton Pest Prediction System",
    page_icon="🌱",
    layout="wide"
)

# === Load Models ===
@st.cache_resource
def load_models():
    """Load all pest prediction models with sklearn version handling"""
    try:
        import joblib
    except ImportError:
        return {}, ["joblib not installed"]
    
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
    
    models = {}
    errors = []
    
    for key, path in model_paths.items():
        try:
            if os.path.exists(path):
                models[key] = joblib.load(path)
            else:
                errors.append(f"{key}: File not found")
        except Exception as e:
            if "_RemainderColsList" in str(e):
                errors.append(f"{key}: Sklearn version mismatch")
            else:
                errors.append(f"{key}: {str(e)}")
    
    return models, errors

def check_sklearn_version():
    """Check sklearn version"""
    try:
        import sklearn
        return sklearn.__version__
    except ImportError:
        return "Not installed"

# === Constants ===
TEHSILS = [
    'haroon abad', 'gojra', 'jhang', 'bhera', 'rojhan', 'kot chutta',
    'faisalabad', 'chicha watni', 'kabir wala', 'sumundri', 'sahiwal',
    'shujabad', 'bhakkar', 'liaquat pur', 'ali pur', 'muzaffargarh',
    'khanewal', 'piplan', 'rahim yar khan', 'arif wala', 'a.p. sial',
    'bahawalnagar', 'kallor kot', 'vehari', 'd.g. khan', 'mianwali',
    'jatoi', 'depalpur', 'multan', 'jahanian', 'shorkot', 'dunya pur',
    'hasilpur', 'chishtian', 'tandlianwala', 'karor lal esan',
    'kehror pacca', 'sadiqabad', 'chobara', 't.t. singh', 'burewala',
    'yazman', 'pak pattan', 'lodhran', 'darya khan', 'taunsa',
    'isa khel', 'renala khurd', 'layyah', 'rajanpur', 'khan pur',
    'minchinabad', 'okara', '18-hazari', 'kot addu', 'a.p.east',
    'mailsi', 'jampur', 'bahawalpur', 'mankera', 'mian channu',
    'quaidabad', 'jalal pur p.w', 'bhowana', 'kasur', 'fort abbas',
    'kot momin', 'sargodha', 'khushab', 'chak jhumra', 'pir mahal',
    'sillanwali', 'chunian', 'patoki', 'jaranwala', 'bhalwal',
    'shahpur', 'kot radha kishan'
]

BASE_TEMP = 10
MAX_TEMP = 30

# === Utility Functions ===
def calculate_gdd(temp_mean, temp_max, temp_min):
    return round(max(min(temp_mean, MAX_TEMP) - BASE_TEMP, 0), 2)

def predict_pests_real(models, inputs):
    """Real prediction using loaded models"""
    daily_gdd = calculate_gdd(inputs['temp_mean'], inputs['temp_max'], inputs['temp_min'])
    weekly_gdd = daily_gdd * 7
    cumulative_gdd = daily_gdd * (inputs['week'] + (inputs['month'] - 1) * 4)
    
    tehsil_encoded = TEHSILS.index(inputs['tehsil']) if inputs['tehsil'] in TEHSILS else 0
    
    input_data = pd.DataFrame([[
        inputs['week'], inputs['month'], inputs['year'], tehsil_encoded,
        inputs['spots_visited'], inputs['area_visited'],
        inputs['temp_mean'], inputs['temp_max'], inputs['temp_min'], inputs['dew_point'],
        daily_gdd, weekly_gdd, cumulative_gdd,
        inputs['nitrogen'], inputs['phosphorus'], inputs['potassium'],
        inputs['humidity'], inputs['ph'], inputs['rainfall']
    ]], columns=[
        'week', 'month', 'year', 'TEHSILS',
        'TOTAL SPOTS VISITED', 'TOTAL AREA VISITED',
        'temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min', 'dew_point_2m_mean',
        'daily_gdd', 'weekly_gdd', 'cumulative_gdd',
        'N', 'P', 'K', 'humidity', 'ph', 'rainfall'
    ])
    
    predictions = {}
    for key, model in models.items():
        try:
            pred = int(model.predict(input_data)[0])
            predictions[key] = pred
        except Exception as e:
            predictions[key] = f"Error: {str(e)}"
    
    return predictions, daily_gdd, weekly_gdd, cumulative_gdd

def predict_pests_demo(inputs):
    """Demo prediction using rule-based logic"""
    daily_gdd = calculate_gdd(inputs['temp_mean'], inputs['temp_max'], inputs['temp_min'])
    weekly_gdd = daily_gdd * 7
    cumulative_gdd = daily_gdd * (inputs['week'] + (inputs['month'] - 1) * 4)
    
    # Simple rule-based predictions for demonstration
    temp_high = inputs['temp_mean'] > 30
    humidity_high = inputs['humidity'] > 70
    rainfall_low = inputs['rainfall'] < 100
    gdd_high = daily_gdd > 15
    
    predictions = {
        "W_FLY": 1 if temp_high and humidity_high else 0,
        "JASSID": 1 if temp_high and rainfall_low else 0,
        "THRIPS": 1 if humidity_high and gdd_high else 0,
        "MBUG": 1 if temp_high else 0,
        "MITES": 1 if rainfall_low and temp_high else 0,
        "APHIDS": 1 if humidity_high else 0,
        "DUSKY": 1 if gdd_high and rainfall_low else 0,
        "SBW": 1 if temp_high and humidity_high else 0,
        "PBW": 1 if gdd_high else 0,
        "ABW": 1 if temp_high and rainfall_low else 0,
        "ARMYWORM": 1 if humidity_high and temp_high else 0,
    }
    
    return predictions, daily_gdd, weekly_gdd, cumulative_gdd

# === Main App ===
def main():
    st.title("🌱 Cotton Pest Prediction System")
    
    # Check sklearn version
    sklearn_version = check_sklearn_version()
    
    # Load models
    models, errors = load_models()
    
    # Show status
    if errors and "_RemainderColsList" in str(errors[0]):
        st.error("⚠️ Sklearn Version Compatibility Issue Detected!")
        
        with st.expander("🔧 How to Fix This Issue", expanded=True):
            st.markdown(f"""
            **Current sklearn version:** {sklearn_version}
            
            **The models were saved with a different sklearn version. Try these solutions:**
            
            1. **Install compatible sklearn version:**
            ```bash
            pip install scikit-learn==1.3.0
            ```
            
            2. **If that doesn't work, try:**
            ```bash
            pip install scikit-learn==1.2.2
            ```
            
            3. **Or try:**
            ```bash
            pip install scikit-learn==1.1.3
            ```
            
            4. **After installing, restart the app:**
            ```bash
            streamlit run dashboard/main.py
            ```
            """)
        
        st.info("🎯 **Demo Mode Active** - Using rule-based predictions for demonstration")
        use_demo_mode = True
        
    elif errors:
        st.error("Model loading errors:")
        for error in errors:
            st.write(f"- {error}")
        use_demo_mode = True
    else:
        st.success(f"✅ Successfully loaded {len(models)} models")
        use_demo_mode = False
    
    # Input form
    st.subheader("📊 Input Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        week = st.number_input("Week (1-4)", min_value=1, max_value=4, value=2)
        month = st.number_input("Month (6-8)", min_value=6, max_value=8, value=7)
        year = st.number_input("Year", min_value=2015, max_value=2030, value=2024)
        tehsil = st.selectbox("Tehsil", options=TEHSILS, index=TEHSILS.index("faisalabad"))
        spots_visited = st.number_input("Total Spots Visited", min_value=1, value=15)
        area_visited = st.number_input("Total Area Visited (acres)", min_value=0.1, value=50.0)
        temp_mean = st.number_input("Mean Temperature (°C)", value=28.5)
        temp_max = st.number_input("Max Temperature (°C)", value=35.0)
        temp_min = st.number_input("Min Temperature (°C)", value=22.0)
        dew_point = st.number_input("Dew Point (°C)", value=18.0)
    
    with col2:
        rainfall = st.number_input("Rainfall (mm)", min_value=0, value=150)
        humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=70)
        nitrogen = st.number_input("Nitrogen (N) ppm", min_value=0, value=120)
        phosphorus = st.number_input("Phosphorus (P) ppm", min_value=0, value=45)
        potassium = st.number_input("Potassium (K) ppm", min_value=0, value=200)
        ph = st.slider("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
    
    # Random inputs button
    if st.button("🎲 Generate Random Inputs"):
        st.rerun()
    
    # Predict button
    button_text = "🔍 Predict Pest Presence (Demo Mode)" if use_demo_mode else "🔍 Predict Pest Presence"
    
    if st.button(button_text):
        inputs = {
            'week': week, 'month': month, 'year': year, 'tehsil': tehsil,
            'spots_visited': spots_visited, 'area_visited': area_visited,
            'temp_mean': temp_mean, 'temp_max': temp_max, 'temp_min': temp_min,
            'dew_point': dew_point, 'rainfall': rainfall, 'humidity': humidity,
            'nitrogen': nitrogen, 'phosphorus': phosphorus, 'potassium': potassium, 'ph': ph
        }
        
        # Choose prediction method
        if use_demo_mode:
            predictions, daily_gdd, weekly_gdd, cumulative_gdd = predict_pests_demo(inputs)
        else:
            predictions, daily_gdd, weekly_gdd, cumulative_gdd = predict_pests_real(models, inputs)
        
        # Results
        st.subheader("🐛 Prediction Results")
        
        if use_demo_mode:
            st.info("ℹ️ These are demo predictions based on simple rules, not actual ML models")
        
        pest_names = {
            "W_FLY": "White Fly",
            "JASSID": "Jassid", 
            "THRIPS": "Thrips",
            "MBUG": "Mirid Bug",
            "MITES": "Mites",
            "APHIDS": "Aphids",
            "DUSKY": "Dusky Cotton Bug",
            "SBW": "Spotted Bollworm",
            "PBW": "Pink Bollworm",
            "ABW": "American Bollworm",
            "ARMYWORM": "Army Worm"
        }
        
        # Count results
        present_count = sum(1 for pred in predictions.values() if pred == 1)
        total_count = len([pred for pred in predictions.values() if not isinstance(pred, str)])
        
        st.write(f"**Summary: {present_count} out of {total_count} pests predicted present**")
        
        # Display predictions in columns
        cols = st.columns(3)
        for i, (key, prediction) in enumerate(predictions.items()):
            with cols[i % 3]:
                if isinstance(prediction, str):
                    st.write(f"❌ **{pest_names[key]}**: {prediction}")
                elif prediction == 1:
                    st.write(f"🔴 **{pest_names[key]}**: PRESENT")
                else:
                    st.write(f"🟢 **{pest_names[key]}**: ABSENT")
        
        # GDD values
        st.subheader("🌡️ Growing Degree Days")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Daily GDD", f"{daily_gdd}°C")
        with col2:
            st.metric("Weekly GDD", f"{weekly_gdd}°C")
        with col3:
            st.metric("Cumulative GDD", f"{cumulative_gdd}°C")

if __name__ == "__main__":
    main()