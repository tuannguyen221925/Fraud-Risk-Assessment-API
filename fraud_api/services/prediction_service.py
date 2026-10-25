import joblib
import json
import pandas as pd
import numpy as np
import shap
from pathlib import Path

BASE_DIR = Path (__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "best_xgb_tuned.pkl"
THRESHOLD_PATH = MODELS_DIR / "best_threshold.json"
MODEL_INFO_PATH = MODELS_DIR / "model_info.json"

model = joblib.load(MODEL_PATH)
explainer = shap.TreeExplainer(model)

#threshold
with open(
    THRESHOLD_PATH,
    "r"
) as f:
    threshold_info = json.load(f)
    
threshold = threshold_info["best_threshold"]

#model info
with open(
    MODEL_INFO_PATH,
    "r"
)   as f:
    model_info = json.load(f)
    

#FEATURE
FEATURE_ORDER = (
    [f"V{i}" for i in range(1, 29)]
    + ["scaled_amount", "scaled_time"]
)

def _prepare_features(
    data: dict
) -> pd.DataFrame:
    df = pd.DataFrame([data])
    return df[FEATURE_ORDER]

#make_prediction
def _make_prediction(
    df: pd.DataFrame
) -> dict:
    probability = float(model.predict_proba(df)[0][1])
    
    prediction = (
        "Fraud"
        if probability >= threshold
        else "Normal"
    )
    
    return {
        "prediction": prediction,
        "probability": round(probability, 4),
    }
    

#predict_transaction
def predict_transaction(
    data: dict 
) -> dict :
    df = _prepare_features(data)
    return _make_prediction(df)

#explain_transaction
def explain_transaction(
    data: dict 
) -> dict:
    df = _prepare_features(data)
    
    prediction_result = _make_prediction (df)
    
    shap_values = explainer.shap_values(df)
    impacts = shap_values[0]
    
    top_indices = np.argsort (
        np.abs(impacts)
    )[::-1][:3]
    
    top_reasons = [
        {
            "feature": FEATURE_ORDER[index],
            "impact":  round(float(impacts[index]), 4),
        }
            for index in top_indices
    ]
    
    return {
        **prediction_result,
        "top_reasons": top_reasons,
    }
    
#model info
def get_model_info() -> dict:
    return model_info