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
    x: List[float]
    
    @validator('x')
    def validate_x_length(cls, v):
        if len(v) != 34:
            raise ValueError(f'Expected 34 features, got {len(v)}')
        return v


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
    Accepts JSON with 'x' field containing a list of 34 numeric values
    """
    try:
        x = pd.DataFrame([data.x])
        
        predictions = predict_logreg(x)
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
    Accepts JSON with 'x' field containing a list of 34 numeric values
    """
    try:
        x = pd.DataFrame([data.x])
        
        predictions = predict_lightgbm(x)
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
    Accepts JSON with 'x' field containing a list of 34 numeric values
    """
    try:
        x = pd.DataFrame([data.x])
        
        predictions = predict_ffnn(x)
        return {"prediction": int(predictions[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)