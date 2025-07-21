import plotly.express as px
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# === Load Models (XGBoost only, not pipelines) ===
model_paths = {
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

# Load all models into dictionary
models = {target: joblib.load(path) for target, path in model_paths.items()}

def fast_shap_explanation(model, X_test, target_name=""):
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
        fig.show()

    except Exception as e:
        print(f"❌ SHAP Error for {target_name}: {str(e)}")
        
                
# === Main ===
if __name__ == "__main__":
    TARGETS = list(model_paths.keys())

    df = pd.read_csv('./data/merged_data (2).csv')
    print(f"✅ Original dataset shape: {df.shape}")
    df = df[:5000]
    print(df.head())

    for target in TARGETS:
        if target not in df.columns:
            print(f"⛔ Skipping target '{target}': not in dataframe")
            continue

        print(f"\n🔍 === Processing Target: {target} ===")
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

        if target in models:
            model = models[target]
            print(f"📈 Generating SHAP explanation for {target}")
            fast_shap_explanation(model, X_test, target_name=target)
        else:
            print(f"❌ Model for {target} not found in model dictionary")
