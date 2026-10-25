import os
import sys
import fastapi
from fastapi import FastAPI
from routers.fraud_router import router 
from schemas.transaction_schema import PredictionRequest
import shap

app = FastAPI(
    title="Fraud Detection API",
    version="1.0",
)


app.include_router(router)

@app.get("/")
def root():
    return {
        "message": "welcome anyone to the fraud detection API"
    }
    
    