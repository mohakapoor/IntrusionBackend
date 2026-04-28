from fastapi import FastAPI, APIRouter, HTTPException, Depends, Security, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
from typing import Dict, Any, List
import pandas as pd
import os
import asyncio
from utils import sampler, len_df
from dotenv import load_dotenv
import predict
from cors import configure_cors


load_dotenv()
app = FastAPI(title="Intrusion Detection API", version="1.0.0")
configure_cors(app)
router = APIRouter(prefix="/intrusiondetection")

class PredictionInput(BaseModel):
    target_class: int

class PredictionByIndexInput(BaseModel):
    index: int

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
    try:
        raw_sample,idx = sampler(data.target_class)
        try:
            predictions, lat = predict._predict_logreg(raw_sample)
            return {"target_class": data.target_class,
                    "prediction": int(predictions),
                    "row_index": int(idx),
                    "detection_time": round(lat, 6)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code =500,detail =f"Sampling error : {str(e)}")


@router.post("/predict/lightgbm",tags=["Predict"])
async def predict_lightgbm_endpoint(
    data: PredictionInput,
    token: str = Depends(verify_token)
):

    try:
        raw_sample,idx = sampler(data.target_class)
        try:
            predictions, lat = predict._predict_lightgbm(raw_sample)
            return {"target_class": data.target_class,
                    "prediction": int(predictions),
                    "row_index": int(idx),
                    "detection_time": round(lat, 6)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code =500,detail =f"Sampling error : {str(e)}")

@router.post("/predict/ffnn",tags=["Predict"])
async def predict_ffnn_endpoint(
    data: PredictionInput,
    token: str = Depends(verify_token)
):

    try:
        raw_sample,idx = sampler(data.target_class)
        try:
            predictions, lat = predict._predict_ffnn(raw_sample)
            return {"target_class": data.target_class,
                    "prediction": int(predictions),
                    "row_index": int(idx),
                    "detection_time": round(lat, 6)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code =500,detail =f"Sampling error : {str(e)}")

@router.post("/predict/autoencoder",tags=["Predict"])
async def predict_autoencoder_endpoint(
    data: PredictionInput,
    token: str = Depends(verify_token)
):

    try:
        raw_sample,idx = sampler(data.target_class)
        try:
            predictions, lat = predict._predict_autoencoder(raw_sample)
            return {"target_class": data.target_class,
                    "prediction": int(predictions),
                    "row_index": int(idx),
                    "detection_time": round(lat, 6)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code =500,detail =f"Sampling error : {str(e)}")

@router.post("/predict/isolationForest",tags=["Predict"])
async def predict_isolation_forest_endpoint(
    data: PredictionInput,
    token: str = Depends(verify_token)
):

    try:
        raw_sample,idx = sampler(data.target_class)
        try:
            predictions, lat = predict._predict_isolation_forest(raw_sample)
            return {"target_class": data.target_class,
                    "prediction": int(predictions),
                    "row_index": int(idx),
                    "detection_time": round(lat, 6)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code =500,detail =f"Sampling error : {str(e)}")

@router.post("/predict/xgboost",tags=["Predict"])
async def predict_xgboost_endpoint(
    data: PredictionInput,
    token: str = Depends(verify_token)
):

    try:
        raw_sample,idx = sampler(data.target_class)
        try:
            predictions, lat = predict._predict_xgboost(raw_sample)
            return {"target_class": data.target_class,
                    "prediction": int(predictions),
                    "row_index": int(idx),
                    "detection_time": round(lat, 6)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code =500,detail =f"Sampling error : {str(e)}")

@router.post("/predict/hybrid",tags=["Predict"])
async def predict_hybrid_endpoint(
    data: PredictionInput,
    token: str = Depends(verify_token)
):

    try:
        result = predict.predict_attack(data.target_class)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    

@router.post("/predict/hybrid_by_index",tags=["Predict"])
async def predict_hybrid_by_index_endpoint(
    data: PredictionByIndexInput,
    token: str = Depends(verify_token)
):

    try:
        result = predict.predict_attack_by_idx(data.index)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    

@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        # Expected JSON: {"command": "start", "start_at": 0}
        initial_data = await websocket.receive_json()
        
        if initial_data.get("command") == "start":
            # Token Check
            if initial_data.get("token") != API_TOKEN:
                await websocket.send_json({"error": "Unauthorized"})
                await websocket.close(code=1008)
                return
            
            await websocket.send_json({"status": "Authentication Successful", "message": "Stream starting..."})
                
            start_idx = initial_data.get("start_at", 0)
            batch_size = 100
            total_rows = len_df()
            
            # Statistics tracking
            stats = {
                "total_processed": 0,
                "correct_predictions": 0,
                "actual_counts": {},
                "predicted_counts": {}
            }
            
            for i in range(start_idx, total_rows, batch_size):
                batch_results = []
                end_idx = min(i + batch_size, total_rows)
                
                for idx in range(i, end_idx):
                    result = predict.predict_attack_by_idx(idx)
                    batch_results.append(result)
                    
                    # Update Statistics
                    actual = str(result["target_class"])
                    # Use LightGBM prediction for accuracy tracking
                    predicted = str(result.get("supervised", {}).get("lightgbm", 0)) if result["status"] == "Attack Detected" else "0"
                    
                    stats["total_processed"] += 1
                    if actual == predicted:
                        stats["correct_predictions"] += 1
                    
                    stats["actual_counts"][actual] = stats["actual_counts"].get(actual, 0) + 1
                    stats["predicted_counts"][predicted] = stats["predicted_counts"].get(predicted, 0) + 1
                
                # Send the batch
                await websocket.send_json({
                    "type": "batch",
                    "batch_start": i,
                    "batch_end": end_idx,
                    "data": batch_results
                })
                
                # await asyncio.sleep(0.5)
            
            # Send Final Statistics
            accuracy = stats["correct_predictions"] / stats["total_processed"] if stats["total_processed"] > 0 else 0
            await websocket.send_json({
                "type": "summary",
                "status": "Stream Complete",
                "statistics": {
                    "accuracy": round(accuracy, 4),
                    **stats
                }
            })
            
            # Close connection after finishing the task
            await websocket.close(code=1000)
                
    except WebSocketDisconnect:
        print(f"WebSocket client disconnected.")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        await websocket.close(code=1011)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)