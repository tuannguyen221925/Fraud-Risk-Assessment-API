from pydantic import BaseModel

class PredictionRequest(
    BaseModel
):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    
    scaled_amount: float
    scaled_time: float

    
class PredictionResponse(
    BaseModel
):
    prediction: str
    probability: float
    

#tạo contract
class TopReason(BaseModel):
    feature:    str
    impact:     float
    
class ExplanationResponse(BaseModel):
    prediction: str
    probability: float
    top_reasons: list[TopReason]
    
class ModelInfoResponse(BaseModel):
    algorithm: str
    version:   str
    precision: float
    recall:    float
    f1:        float
    threshold: float