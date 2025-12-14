from fastapi import FastAPI, APIRouter, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
from typing import Dict, Any, List
import pandas as pd
import os
from dotenv import load_dotenv
from predict import predict_ffnn, predict_lightgbm, predict_logreg
from cors import configure_cors


load_dotenv()
app = FastAPI(title="Intrusion Detection API", version="1.0.0")
configure_cors(app)
router = APIRouter(prefix="/intrusiondetection")

class PredictionInput(BaseModel):
    target_class: int


security = HTTPBearer()
API_TOKEN = os.getenv("TOKEN")
def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify the Bearer token"""
    token = credentials.credentials
    if token != API_TOKEN:
        raise HTTPException(
            status_code=403,
            detail="Invalid authentication token"
        )
    return token


# API name endpoint
@router.get("/api_name")
async def api_name():
    return {"api_name": "Intrusion Detection API"}

# Health check endpoint
@router.get("/health")
async def health():
    return {"status": "healthy"}


# Prediction endpoints
@router.post("/predict/logreg",tags=["Predict"])
async def predict_logreg_endpoint(
    data: PredictionInput,
    token: str = Depends(verify_token)
):
    """
    Logistic Regression prediction endpoint
    Accepts JSON with 'target_class' (int)
    """
    try:
        predictions = predict_logreg(data.target_class)
        return {"prediction": int(predictions[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/predict/lightgbm",tags=["Predict"])
async def predict_lightgbm_endpoint(
    data: PredictionInput,
    token: str = Depends(verify_token)
):
    """
    LightGBM prediction endpoint
    Accepts JSON with 'target_class' (int)
    """
    try:
        predictions = predict_lightgbm(data.target_class)
        return {"prediction": int(predictions[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/predict/ffnn",tags=["Predict"])
async def predict_ffnn_endpoint(
    data: PredictionInput,
    token: str = Depends(verify_token)
):
    """
    Feed-forward Neural Network prediction endpoint
    Accepts JSON with 'target_class' (int)
    """
    try:
        predictions = predict_ffnn(data.target_class)
        return {"prediction": int(predictions[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)