import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

PIPELINE_PATH = MODELS_DIR / "fraud_pipeline.pkl"
THRESHOLD_PATH = MODELS_DIR / "pipeline_threshold.json"
MODEL_INFO_PATH = MODELS_DIR / "pipeline_model_info.json"

# Pipeline gồm preprocessor và XGBoost classifier
pipeline = joblib.load(PIPELINE_PATH)
classifier = pipeline.named_steps["model"]
classifier.get_booster().set_param({"device": "cpu"})
preprocessor = pipeline.named_steps["preprocessor"]

# SHAP giải thích XGBoost, còn predict dùng toàn bộ pipeline
explainer = shap.TreeExplainer(classifier)

with open(THRESHOLD_PATH, "r") as file:
    threshold_info = json.load(file)

threshold = threshold_info["best_threshold"]

with open(MODEL_INFO_PATH, "r") as file:
    model_info = json.load(file)


RAW_FEATURE_ORDER = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)


def _prepare_features(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    return df[RAW_FEATURE_ORDER]


def _make_prediction(df: pd.DataFrame) -> dict:
    # Pipeline tự scale Amount/Time rồi gọi XGBoost
    probability = float(pipeline.predict_proba(df)[0][1])

    prediction = (
        "Fraud"
        if probability >= threshold
        else "Normal"
    )

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
    }


def predict_transaction(data: dict) -> dict:
    df = _prepare_features(data)
    return _make_prediction(df)


def explain_transaction(data: dict) -> dict:
    df = _prepare_features(data)
    prediction_result = _make_prediction(df)

    # SHAP cần đúng dữ liệu đã qua bước preprocessing
    transformed_df = preprocessor.transform(df)
    shap_values = explainer.shap_values(transformed_df)
    impacts = np.asarray(shap_values)[0]

    feature_names = list(transformed_df.columns)
    top_indices = np.argsort(np.abs(impacts))[::-1][:3]

    top_reasons = [
        {
            "feature": feature_names[index],
            "impact": round(float(impacts[index]), 4),
        }
        for index in top_indices
    ]

    return {
        **prediction_result,
        "top_reasons": top_reasons,
    }


def get_model_info() -> dict:
    return model_info