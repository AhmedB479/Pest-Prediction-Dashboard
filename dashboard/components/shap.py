import plotly.express as px
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
import os

class ShapExplainer:
    """SHAP explanation component for pest prediction models"""
    
    def __init__(self):
        # === Load Models (XGBoost only, not pipelines) ===
        self.model_paths = {
            "W. FLY (%)": './models/tuned_models/xgboost_W_FLY_model.pkl',
            "JASSID (%)": './models/tuned_models/xgboost_JASSID_model.pkl',
            "THRIPS (%)": './models/tuned_models/xgboost_THRIPS_model.pkl',
            "M.BUG (%)": './models/tuned_models/xgboost_M_BUG_model.pkl',
            "MITES (%)": './models/tuned_models/xgboost_MITES_model.pkl',
            "APHIDS (%)": './models/tuned_models/xgboost_APHIDS_model.pkl',
            "DUSKY COTTON BUG (%)": './models/tuned_models/xgboost_DUSKY_COTTON_BUG_model.pkl',
            "SBW (%)": './models/tuned_models/xgboost_SBW_model.pkl',
            "PBW (%)": './models/tuned_models/xgboost_PBW_model.pkl',
            "ABW (%)": './models/tuned_models/xgboost_ABW_model.pkl',
            "Army Worm (%)": './models/tuned_models/xgboost_Army_Worm_model.pkl',
        }
        
        self.models = None
        self._load_models()
    
    def _load_models(self):
        """Load all models into dictionary"""
        self.models = {}
        for target, path in self.model_paths.items():
            try:
                if os.path.exists(path):
                    self.models[target] = joblib.load(path)
                else:
                    st.warning(f"Model file not found: {path}")
            except Exception as e:
                st.error(f"Error loading model for {target}: {e}")
    
    def fast_shap_explanation(self, model, X_test, target_name=""):
        """Generate SHAP explanation - keeping original logic intact"""
        try:
            pipeline = model
            preprocessor = pipeline.named_steps['preprocess']
            classifier = pipeline.named_steps['model']

            # Transform input
            X_transformed = preprocessor.transform(X_test)

            # Use TreeExplainer (works with XGBoost, RandomForest, DecisionTree, AdaBoost)
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_transformed)

            # Get feature names
            try:
                feature_names = preprocessor.get_feature_names_out()
            except:
                feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

            # Mean absolute SHAP values
            if isinstance(shap_values, list):  # Multi-class
                shap_values = shap_values[1]  # Class 1 (positive class)

            mean_abs_shap = np.abs(shap_values).mean(axis=0)

            # Plot
            df_shap = pd.DataFrame({
                "Feature": feature_names,
                "Mean |SHAP value|": mean_abs_shap
            }).sort_values("Mean |SHAP value|", ascending=False)

            fig = px.bar(
                df_shap,
                x="Mean |SHAP value|",
                y="Feature",
                orientation="h",
                title=f"SHAP Feature Importance for {target_name}",
                height=600
            )
            fig.update_layout(yaxis=dict(autorange="reversed"))
            
            return fig

        except Exception as e:
            st.error(f"❌ SHAP Error for {target_name}: {str(e)}")
            return None
    
    def load_and_process_data(self, data_path='./data/merged_data (2).csv', max_samples=5000):
        """Load and process data for SHAP analysis"""
        try:
            df = pd.read_csv(data_path)
            st.success(f"✅ Original dataset shape: {df.shape}")
            df = df[:max_samples]
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    
    def generate_all_shap_explanations(self, df):
        """Generate SHAP explanations for all targets"""
        if self.models is None or len(self.models) == 0:
            st.warning("No models loaded. Please check model files.")
            return {}
        
        TARGETS = list(self.model_paths.keys())
        explanations = {}
        
        for target in TARGETS:
            if target not in df.columns:
                st.warning(f"⛔ Skipping target '{target}': not in dataframe")
                continue

            st.write(f"🔍 Processing Target: {target}")
            current_df = df.dropna(subset=[target])
            X = current_df.drop(columns=TARGETS)
            y = current_df[target]

            # Split data (handle stratify errors for regression or imbalanced targets)
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
                )
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, shuffle=True
                )

            if target in self.models:
                model = self.models[target]
                st.write(f"📈 Generating SHAP explanation for {target}")
                fig = self.fast_shap_explanation(model, X_test, target_name=target)
                if fig:
                    explanations[target] = fig
            else:
                st.error(f"❌ Model for {target} not found in model dictionary")
        
        return explanations
    
    def display_shap_ui(self):
        """Display SHAP analysis UI in Streamlit"""
        st.title("📈 SHAP Model Analysis")
        st.markdown("---")
        
        st.markdown("""
        ## SHapley Additive exPlanations (SHAP)
        
        SHAP values provide a unified approach to explaining model predictions by computing 
        the contribution of each feature to the prediction based on cooperative game theory.
        """)
        
        # Data loading section
        st.subheader("📊 Data Loading")
        
        col1, col2 = st.columns(2)
        with col1:
            max_samples = st.number_input("Max samples to use", min_value=100, max_value=10000, value=5000)
        with col2:
            data_path = st.text_input("Data file path", value="./data/merged_data (2).csv")
        
        if st.button("📁 Load Data"):
            df = self.load_and_process_data(data_path, max_samples)
            if df is not None:
                st.session_state['shap_data'] = df
                with st.expander("View Data Sample"):
                    st.dataframe(df.head())
        
        # SHAP analysis section
        if 'shap_data' in st.session_state:
            st.subheader("🎯 Generate SHAP Analysis")
            
            if st.button("🔍 Generate SHAP Explanations"):
                with st.spinner("Generating SHAP explanations... This may take a while."):
                    explanations = self.generate_all_shap_explanations(st.session_state['shap_data'])
                
                if explanations:
                    st.success(f"✅ Generated SHAP explanations for {len(explanations)} models")
                    
                    # Display explanations
                    for target, fig in explanations.items():
                        st.subheader(f"🐛 {target}")
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown("---")
                else:
                    st.error("No SHAP explanations generated. Please check your models and data.")
        else:
            st.info("Please load data first to generate SHAP explanations.")
        
        # How SHAP works section
        with st.expander("📖 How SHAP Works"):
            st.markdown("""
            1. **Shapley Values**: Based on cooperative game theory principles
            2. **Feature Contributions**: Calculate how much each feature contributes to the prediction
            3. **Additive Property**: The sum of SHAP values equals the difference from expected value
            4. **Global & Local**: Provides both instance-level and global feature importance
            
            **Benefits:**
            - Theoretically grounded and mathematically sound
            - Consistent and fair feature attributions
            - Multiple visualization types available
            - Works with tree-based models efficiently
            - Provides global feature importance rankings
            
            **Note:** This analysis uses TreeExplainer which is optimized for tree-based models like XGBoost.
            """)
        
        # Model information
        with st.expander("🤖 Loaded Models"):
            if self.models:
                for target, model in self.models.items():
                    st.write(f"✅ {target}: XGBoost model loaded")
            else:
                st.write("❌ No models loaded")

# Create a global instance for easy import
shap_explainer = ShapExplainer()

# Convenience functions for easy import
def generate_shap_explanation(model, X_test, target_name):
    """Convenience function to generate a single SHAP explanation"""
    return shap_explainer.fast_shap_explanation(model, X_test, target_name)

def generate_all_shap_explanations(df):
    """Convenience function to generate all SHAP explanations"""
    return shap_explainer.generate_all_shap_explanations(df)

def show_shap_ui():
    """Convenience function to display SHAP UI"""
    shap_explainer.display_shap_ui()
