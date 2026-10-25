from fastapi import APIRouter
from schemas.transaction_schema import (
    PredictionRequest,
    PredictionResponse,
    ExplanationResponse,
    ModelInfoResponse,
)
from services.prediction_service import (
    predict_transaction,
    explain_transaction,
    get_model_info,
)


router = APIRouter(
    prefix="/fraud",
    tags=["Fraud"],
)

@router.get("/")
def health_check():
    return {
        "status": "healthy"
    }
    
@router.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    payload: PredictionRequest
):

    return predict_transaction(
        payload.model_dump()
    )
@router.post(
    "/explain",
    response_model=ExplanationResponse,
)
def explain(
    payload: PredictionRequest,
):
    return explain_transaction(
        payload.model_dump()
    )
    
#model info
@router.get(
    "/model-info",
    response_model= ModelInfoResponse,
)
def model_info():
        return get_model_info()