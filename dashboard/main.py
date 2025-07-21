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

# === Sidebar Navigation ===
def create_sidebar():
    """Create sidebar navigation"""
    st.sidebar.title("🌱 Navigation")
    st.sidebar.markdown("---")
    
    # Navigation options
    pages = {
        "🏠 Home": "home",
        "🔍 Pest Prediction": "prediction", 
        "📊 LIME Explanations": "lime",
        "📈 SHAP Analysis": "shap",
        "📋 Model Information": "model_info",
        "ℹ️ About": "about"
    }
    
    # Create radio buttons for navigation
    selected_page = st.sidebar.radio(
        "Select a page:",
        list(pages.keys()),
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌾 Cotton Pest Dashboard")
    st.sidebar.markdown("Advanced ML-based pest prediction system")
    
    return pages[selected_page]

# === Load Models ===
@st.cache_resource
def load_models():
    """Load all pest prediction models with sklearn version handling"""
    try:
        import joblib
    except ImportError:
        return {}, ["joblib not installed"]
    
    model_paths = {
        "W_FLY": './models/tuned_models/xgboost_W_FLY_model.pkl',
        "JASSID": './models/tuned_models/xgboost_JASSID_model.pkl',
        "THRIPS": './models/tuned_models/stacking_THRIPS_model.pkl',
        "MBUG": './models/tuned_models/stacking_M_BUG_model.pkl',
        "MITES": './models/tuned_models/random_forest_MITES_model.pkl',
        "APHIDS": './models/tuned_models/random_forest_APHIDS_model.pkl',
        "DUSKY": './models/tuned_models/stacking_DUSKY_COTTON_BUG_model.pkl',
        "SBW": './models/tuned_models/random_forest_SBW_model.pkl',
        "PBW": './models/tuned_models/stacking_PBW_model.pkl',
        "ABW": './models/tuned_models/knn_ABW_model.pkl',
        "ARMYWORM": './models/tuned_models/knn_Army_Worm_model.pkl',
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
            elif "numpy._core" in str(e):
                errors.append(f"{key}: Numpy version mismatch")
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

def check_numpy_version():
    """Check numpy version"""
    try:
        import numpy
        return numpy.__version__
    except ImportError:
        return "Not installed"

def generate_random_inputs():
    """Generate random input values for testing"""
    return {
        'week': random.randint(1, 4),
        'month': random.randint(6, 8),  # Cotton season
        'year': random.randint(2020, 2024),
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

# === Page Functions ===
def show_home_page():
    """Home/Landing page"""
    st.title("🌱 Cotton Pest Prediction System")
    st.markdown("---")
    
    # Hero section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to the Advanced Cotton Pest Prediction Dashboard
        
        This intelligent system uses machine learning models to predict the presence of various cotton pests 
        based on environmental and agricultural parameters. Make informed decisions to protect your cotton crops!
        
        ### 🎯 Key Features:
        - **11 Pest Types**: Predict presence of major cotton pests
        - **ML Models**: Advanced ensemble methods (Stacking, Voting, XGBoost, etc.)
        - **Real-time Analysis**: Instant predictions with environmental data
        - **Model Explainability**: LIME and SHAP analysis for transparency
        - **Growing Degree Days**: Temperature-based crop development tracking
        """)
        
    with col2:
        st.markdown("""
        ### 🐛 Predicted Pests:
        - White Fly
        - Jassid
        - Thrips
        - Mirid Bug
        - Mites
        - Aphids
        - Dusky Cotton Bug
        - Spotted Bollworm
        - Pink Bollworm
        - American Bollworm
        - Army Worm
        """)
    
    st.markdown("---")
    
    # Quick stats
    st.subheader("📊 System Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Pest Types", "11")
    with col2:
        st.metric("🤖 ML Models", "9 Types")
    with col3:
        st.metric("🌍 Tehsils Covered", f"{len(TEHSILS)}")
    with col4:
        st.metric("📈 Parameters", "16")
    
    st.markdown("---")
    
    # Getting started
    st.subheader("🚀 Getting Started")
    st.markdown("""
    1. **Navigate to Pest Prediction** to start making predictions
    2. **Input environmental parameters** like temperature, humidity, rainfall
    3. **Add soil parameters** like N-P-K values and pH
    4. **Get instant predictions** for all 11 pest types
    5. **Explore explanations** using LIME and SHAP analysis
    """)
    
    # Version info
    with st.expander("ℹ️ System Information"):
        sklearn_version = check_sklearn_version()
        numpy_version = check_numpy_version()
        
        st.markdown(f"""
        **System Versions:**
        - Scikit-learn: {sklearn_version}
        - NumPy: {numpy_version}
        - Streamlit: {st.__version__}
        
        **Model Information:**
        - Total Models: 99 trained models (9 algorithms × 11 pests)
        - Best performing models selected automatically
        - Cross-validated and hyperparameter tuned
        """)

def show_prediction_page():
    """Pest prediction page - original main functionality"""
    st.title("🔍 Cotton Pest Prediction")
    st.markdown("---")
    
    # Check versions and load models
    sklearn_version = check_sklearn_version()
    numpy_version = check_numpy_version()
    
    # Load models
    models, errors = load_models()
    
    # Show status
    if errors and ("numpy._core" in str(errors[0]) or "No module named 'numpy._core'" in str(errors[0])):
        st.error("⚠️ Numpy Version Compatibility Issue Detected!")
        
        with st.expander("🔧 How to Fix This Issue", expanded=True):
            st.markdown(f"""
            **Current versions:**
            - Sklearn: {sklearn_version}
            - Numpy: {numpy_version}
            
            **The models were saved with numpy 1.x but you're using numpy 2.x. Try:**
            
            ```bash
            pip uninstall numpy -y
            pip install numpy==1.26.4
            ```
            
            **Then restart the app:**
            ```bash
            streamlit run dashboard/main.py
            ```
            """)
        
        st.info("🎯 **Demo Mode Active** - Using rule-based predictions for demonstration")
        use_demo_mode = True
        
    elif errors and "_RemainderColsList" in str(errors[0]):
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
    
    # Get values from session state or use defaults
    with col1:
        week = st.number_input("Week (1-4)", min_value=1, max_value=4, 
                               value=st.session_state.get('random_week', 2), key="week")
        month = st.number_input("Month (6-8)", min_value=6, max_value=8, 
                                value=st.session_state.get('random_month', 7), key="month")
        year = st.number_input("Year", min_value=2015, max_value=2030, 
                               value=st.session_state.get('random_year', 2024), key="year")
        
        tehsil_default = st.session_state.get('random_tehsil', 'faisalabad')
        tehsil_index = TEHSILS.index(tehsil_default) if tehsil_default in TEHSILS else TEHSILS.index('faisalabad')
        tehsil = st.selectbox("Tehsil", options=TEHSILS, index=tehsil_index, key="tehsil")
        
        spots_visited = st.number_input("Total Spots Visited", min_value=1, 
                                        value=st.session_state.get('random_spots_visited', 15), key="spots_visited")
        area_visited = st.number_input("Total Area Visited (acres)", min_value=0.1, 
                                       value=st.session_state.get('random_area_visited', 50.0), key="area_visited")
        temp_mean = st.number_input("Mean Temperature (°C)", 
                                    value=st.session_state.get('random_temp_mean', 28.5), key="temp_mean")
        temp_max = st.number_input("Max Temperature (°C)", 
                                   value=st.session_state.get('random_temp_max', 35.0), key="temp_max")
        temp_min = st.number_input("Min Temperature (°C)", 
                                   value=st.session_state.get('random_temp_min', 22.0), key="temp_min")
        dew_point = st.number_input("Dew Point (°C)", 
                                    value=st.session_state.get('random_dew_point', 18.0), key="dew_point")
    
    with col2:
        rainfall = st.number_input("Rainfall (mm)", min_value=0, 
                                   value=st.session_state.get('random_rainfall', 150), key="rainfall")
        humidity = st.slider("Humidity (%)", min_value=0, max_value=100, 
                             value=st.session_state.get('random_humidity', 70), key="humidity")
        nitrogen = st.number_input("Nitrogen (N) ppm", min_value=0, 
                                   value=st.session_state.get('random_nitrogen', 120), key="nitrogen")
        phosphorus = st.number_input("Phosphorus (P) ppm", min_value=0, 
                                     value=st.session_state.get('random_phosphorus', 45), key="phosphorus")
        potassium = st.number_input("Potassium (K) ppm", min_value=0, 
                                    value=st.session_state.get('random_potassium', 200), key="potassium")
        ph = st.slider("Soil pH", min_value=0.0, max_value=14.0, 
                       value=st.session_state.get('random_ph', 6.5), key="ph")
    
    # Random inputs button - FIXED!
    if st.button("🎲 Generate Random Inputs"):
        # Generate random values
        random_vals = generate_random_inputs()
        
        # Store in session state
        for key, value in random_vals.items():
            st.session_state[f'random_{key}'] = value
        
        # Rerun to update the UI
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

from components.lime import show_lime_ui, lime_explainer

# In your show_lime_page() function:
def show_lime_page():
    """LIME explanations page"""
    show_lime_ui()




# At the top of main.py
from components.shap import show_shap_ui

# In your show_shap_page() function:
def show_shap_page():
    """SHAP analysis page"""
    show_shap_ui()

def show_model_info_page():
    """Model information page"""
    st.title("📋 Model Information")
    st.markdown("---")
    
    st.markdown("""
    ## Machine Learning Models Overview
    
    Our pest prediction system uses multiple state-of-the-art machine learning algorithms
    to ensure robust and accurate predictions.
    """)
    
    # Model types
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Algorithm Types")
        algorithms = [
            "Random Forest", "XGBoost", "K-Nearest Neighbors",
            "Support Vector Machine", "Decision Tree", "AdaBoost",
            "Bagging", "Stacking Ensemble", "Voting Ensemble"
        ]
        
        for algo in algorithms:
            st.write(f"• {algo}")
    
    with col2:
        st.subheader("🎯 Target Pests")
        pests = [
            "White Fly", "Jassid", "Thrips", "Mirid Bug", "Mites",
            "Aphids", "Dusky Cotton Bug", "Spotted Bollworm", 
            "Pink Bollworm", "American Bollworm", "Army Worm"
        ]
        
        for pest in pests:
            st.write(f"• {pest}")
    
    st.markdown("---")
    
    # Features
    st.subheader("📊 Input Features")
    
    feature_categories = {
        "🕒 Temporal": ["Week", "Month", "Year"],
        "🌍 Location": ["Tehsil"],
        "🔍 Survey": ["Total Spots Visited", "Total Area Visited"],
        "🌡️ Weather": ["Mean Temperature", "Max Temperature", "Min Temperature", "Dew Point", "Humidity", "Rainfall"],
        "🌱 Soil": ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)", "pH"],
        "📈 Computed": ["Daily GDD", "Weekly GDD", "Cumulative GDD"]
    }

    st.subheader("All Models Performance")
    for category, features in feature_categories.items():
        with st.expander(f"{category} Features"):
            for feature in features:
                st.write(f"• {feature}")
    
    # Read and display model evaluation results
    try:
        model_results_df = pd.read_csv('./data/model_evaluation_results.csv')
        st.dataframe(model_results_df, use_container_width=True)
    except FileNotFoundError:
        st.error("Model evaluation results file not found at './data/model_evaluation_results.csv'")
    except Exception as e:
        st.error(f"Error loading model evaluation results: {e}")

    # Model selection info
    st.markdown("---")
    st.subheader("🏆 Model Selection")
    st.markdown("""
    **Best Performing Models by Pest:**
    - **White Fly**: XGBoost
    - **Jassid**: XGBoost  
    - **Thrips**: Stacking Ensemble
    - **Mirid Bug**: Stacking Ensemble
    - **Mites**: Random Forest
    - **Aphids**: Random Forest
    - **Dusky Cotton Bug**: Stacking Ensemble
    - **Spotted Bollworm**: Random Forest
    - **Pink Bollworm**: Stacking Ensemble
    - **American Bollworm**: K-Nearest Neighbors
    - **Army Worm**: K-Nearest Neighbors
    
    *Models were selected based on cross-validation performance and hyperparameter tuning.*
    """)

def show_about_page():
    """About page"""
    st.title("ℹ️ About the Cotton Pest Prediction System")
    st.markdown("---")
    
    st.markdown("""
    ## 🌾 Project Overview
    
    The Cotton Pest Prediction System is an advanced machine learning application designed to help
    farmers and agricultural experts predict the presence of various cotton pests based on 
    environmental and agricultural parameters.
    
    ## 🎯 Objectives
    
    - **Early Warning**: Provide early detection of pest presence
    - **Data-Driven Decisions**: Enable informed pest management strategies  
    - **Crop Protection**: Help minimize crop damage and losses
    - **Resource Optimization**: Optimize pesticide use and timing
    
    ## 🔬 Technology Stack
    
    - **Machine Learning**: Scikit-learn, XGBoost
    - **Web Framework**: Streamlit
    - **Data Processing**: Pandas, NumPy
    - **Explainability**: LIME, SHAP
    - **Visualization**: Plotly, Matplotlib
    
    ## 📊 Data & Models
    
    - **Training Data**: Historical pest occurrence data with environmental parameters
    - **Cross-Validation**: Rigorous model validation and selection
    - **Hyperparameter Tuning**: Optimized model parameters for best performance
    - **Ensemble Methods**: Multiple algorithms combined for robust predictions
    
    ## 🌍 Coverage Area
    
    The system covers **{} tehsils** across cotton-growing regions, providing localized
    predictions based on regional characteristics.
    
    ## 📈 Performance
    
    - **99 Total Models**: 9 algorithms × 11 pest types
    - **Cross-Validated**: All models validated using cross-validation
    - **Best Model Selection**: Automatic selection of best-performing model per pest
    - **Continuous Improvement**: Regular model updates and retraining
    
    ## 🤝 Usage Guidelines
    
    1. **Input Accuracy**: Ensure accurate input parameters for reliable predictions
    2. **Regular Monitoring**: Use predictions as part of regular field monitoring
    3. **Expert Consultation**: Combine predictions with expert agricultural advice
    4. **Integrated Approach**: Use as part of comprehensive pest management strategy
    
    ## ⚠️ Disclaimer
    
    This system provides predictions based on historical data and machine learning models.
    While highly accurate, predictions should be used as guidance alongside professional
    agricultural expertise and field observations.
    """.format(len(TEHSILS)))
    
    st.markdown("---")
    st.markdown("*Developed for advancing precision agriculture and sustainable farming practices.*")

# === Main App ===
def main():
    # Create sidebar navigation
    selected_page = create_sidebar()
    
    # Route to appropriate page
    if selected_page == "home":
        show_home_page()
    elif selected_page == "prediction":
        show_prediction_page()
    elif selected_page == "lime":
        show_lime_page()
    elif selected_page == "shap":
        show_shap_page()
    elif selected_page == "model_info":
        show_model_info_page()
    elif selected_page == "about":
        show_about_page()

if __name__ == "__main__":
    main()