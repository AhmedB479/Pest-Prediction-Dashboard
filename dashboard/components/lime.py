import joblib
import pandas as pd
import numpy as np
import lime
import lime.lime_tabular
import plotly.graph_objects as go
import streamlit as st
from sklearn.pipeline import Pipeline
import re
import os

class LimeExplainer:
    """LIME explanation component for pest prediction models"""
    
    def __init__(self):
        self.TARGET = [
            'W. FLY (%)', 'JASSID (%)', 'THRIPS (%)', 'M.BUG (%)', 'MITES (%)',
            'APHIDS (%)', 'DUSKY COTTON BUG (%)', 'SBW (%)', 'PBW (%)', 'ABW (%)',
            'Army Worm (%)'
        ]
        
        self.model_paths = {
            "W_FLY": './models/tuned_models/xgboost_W_FLY_model.pkl',
            "JASSID": './models/tuned_models/xgboost_JASSID_model.pkl',
            "THRIPS": './models/tuned_models/xgboost_THRIPS_model.pkl',
            "MBUG": './models/tuned_models/xgboost_M_BUG_model.pkl',
            "MITES": './models/tuned_models/xgboost_MITES_model.pkl',
            "APHIDS": './models/tuned_models/xgboost_APHIDS_model.pkl',
            "DUSKY": './models/tuned_models/xgboost_DUSKY_COTTON_BUG_model.pkl',
            "SBW": './models/tuned_models/xgboost_SBW_model.pkl',
            "PBW": './models/tuned_models/xgboost_PBW_model.pkl',
            "ABW": './models/tuned_models/xgboost_ABW_model.pkl',
            "ARMYWORM": './models/tuned_models/xgboost_Army_Worm_model.pkl',
        }
        
        self.models = None
        self._load_models()
    
    def _load_models(self):
        """Load all available models"""
        self.models = {}
        
        # Create proper target name mapping
        target_mapping = {
            'W. FLY (%)': 'W_FLY',
            'JASSID (%)': 'JASSID', 
            'THRIPS (%)': 'THRIPS',
            'M.BUG (%)': 'M_BUG',
            'MITES (%)': 'MITES',
            'APHIDS (%)': 'APHIDS',
            'DUSKY COTTON BUG (%)': 'DUSKY_COTTON_BUG',
            'SBW (%)': 'SBW',
            'PBW (%)': 'PBW',
            'ABW (%)': 'ABW',
            'Army Worm (%)': 'Army_Worm'
        }
        
        for target in self.TARGET:
            if target in target_mapping:
                sanitized_target = target_mapping[target]
                for model_type in ['stacking', 'voting', 'random_forest', 'xgboost', 'knn', 'adaboost']:
                    try:
                        model_path = f'./models/tuned_models/{model_type}_{sanitized_target}_model.pkl'
                        if os.path.exists(model_path):
                            model = joblib.load(model_path)
                            self.models[target] = model
                            print(f"✅ Loaded {model_type} model for {target}")
                            break
                    except Exception as e:
                        continue
                
                if target not in self.models:
                    print(f"❌ No model found for {target}")
    
    def generate_explanation(self, model, instance, feature_names, target_name, class_names=['No Pest', 'Pest']):
        """
        Generate LIME explanation for a single prediction
        
        Args:
            model: Trained model pipeline
            instance: Input data (pandas DataFrame)
            feature_names: List of feature names
            target_name: Name of the target pest
            class_names: Class labels for classification
            
        Returns:
            plotly.graph_objects.Figure or None: LIME explanation plot
        """
        try:
            # Extract preprocessor and model from pipeline
            preprocessor = model.named_steps['preprocess']
            predictor = model.named_steps['model']

            # Transform input data
            processed_instance = preprocessor.transform(instance)

            # Get feature names after preprocessing
            if hasattr(preprocessor, 'get_feature_names_out'):
                processed_feature_names = preprocessor.get_feature_names_out()
            else:
                processed_feature_names = feature_names

            # Create LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=processed_instance,
                feature_names=processed_feature_names,
                class_names=class_names,
                mode='classification',
                discretize_continuous=True
            )

            # Generate explanation for first instance
            exp = explainer.explain_instance(
                processed_instance[0],
                predictor.predict_proba,
                num_features=10
            )

            # Convert LIME explanation to Plotly figure
            lime_list = exp.as_list()
            features = [x[0] for x in lime_list]
            values = [x[1] for x in lime_list]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=values,
                y=features,
                orientation='h',
                marker_color=['green' if x > 0 else 'red' for x in values],
                text=[f'{v:.3f}' for v in values],
                textposition='auto'
            ))

            fig.update_layout(
                title=f'LIME Explanation for {target_name}',
                xaxis_title='Feature Importance',
                yaxis_title='Features',
                height=600,
                showlegend=False,
                yaxis=dict(autorange="reversed")
            )
            
            return fig

        except Exception as e:
            st.error(f"LIME Error for {target_name}: {str(e)}")
            return None
    
    def generate_all_explanations(self, input_data, feature_names):
        """
        Generate LIME explanations for all available models
        
        Args:
            input_data: Input data (pandas DataFrame)
            feature_names: List of feature names
            
        Returns:
            dict: Dictionary of target names and their corresponding figures
        """
        explanations = {}
        
        if self.models is None or len(self.models) == 0:
            st.warning("No models loaded. Please check model files.")
            return explanations
        
        for target in self.TARGET:
            if target in self.models:
                fig = self.generate_explanation(
                    model=self.models[target],
                    instance=input_data,
                    feature_names=feature_names,
                    target_name=target
                )
                if fig:
                    explanations[target] = fig
            else:
                st.warning(f"Model for {target} not found")
        
        return explanations
    
    def load_sample_data(self, data_path='./data/merged_data (2).csv', n_samples=100):
        """
        Load sample data for testing LIME explanations
        
        Args:
            data_path: Path to the data file
            n_samples: Number of samples to load
            
        Returns:
            tuple: (sample_data, feature_names) or (None, None) if error
        """
        try:
            df = pd.read_csv(data_path)
            # Get feature columns (excluding target columns)
            feature_cols = [col for col in df.columns if '(%)' not in col]
            sample_data = df[feature_cols].iloc[:n_samples]
            return sample_data, feature_cols
        except Exception as e:
            st.error(f"Error loading sample data: {e}")
            return None, None
    
    def display_explanations_ui(self):
        """
        Display LIME explanations UI in Streamlit
        """
        st.title("📊 LIME Model Explanations")
        st.markdown("---")
        
        st.markdown("""
        ## Local Interpretable Model-agnostic Explanations (LIME)
        
        LIME explains individual predictions by learning an interpretable model locally around the prediction.
        It helps understand which features contributed most to a specific prediction.
        """)
        
        # Load sample data
        sample_data, feature_names = self.load_sample_data()
        
        if sample_data is not None:
            st.subheader("🎯 Generate LIME Explanations")
            
            # Option to use sample data or upload custom data
            use_sample = st.checkbox("Use sample data", value=True)
            
            if use_sample:
                st.info("Using sample data from the dataset for demonstration")
                input_data = sample_data
                
                # Show sample data
                with st.expander("View Sample Data"):
                    st.dataframe(sample_data)
                
            else:
                st.info("Custom data upload functionality can be added here")
                input_data = sample_data  # Fallback to sample data
            
            # Generate explanations button
            if st.button("🔍 Generate LIME Explanations"):
                with st.spinner("Generating LIME explanations..."):
                    explanations = self.generate_all_explanations(input_data, feature_names)
                
                if explanations:
                    st.success(f"✅ Generated explanations for {len(explanations)} models")
                    
                    # Display explanations
                    for target, fig in explanations.items():
                        st.subheader(f"🐛 {target}")
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown("---")
                else:
                    st.error("No explanations generated. Please check your models and data.")
        
        else:
            st.error("Unable to load sample data. Please check the data file path.")
        
        # How LIME works section
        with st.expander("📖 How LIME Works"):
            st.markdown("""
            1. **Local Sampling**: Generate samples around the instance to explain
            2. **Model Predictions**: Get predictions for these samples  
            3. **Learn Simple Model**: Fit an interpretable model (e.g., linear) to these samples
            4. **Extract Explanations**: Use the simple model to explain the prediction
            
            **Benefits:**
            - Model-agnostic (works with any ML model)
            - Instance-specific explanations
            - Easy to understand visualizations
            - Shows positive and negative feature contributions
            """)

# Create a global instance for easy import
lime_explainer = LimeExplainer()

# Convenience functions for easy import
def generate_lime_explanation(model, instance, feature_names, target_name):
    """Convenience function to generate a single LIME explanation"""
    return lime_explainer.generate_explanation(model, instance, feature_names, target_name)

def generate_all_lime_explanations(input_data, feature_names):
    """Convenience function to generate all LIME explanations"""
    return lime_explainer.generate_all_explanations(input_data, feature_names)

def show_lime_ui():
    """Convenience function to display LIME UI"""
    lime_explainer.display_explanations_ui()