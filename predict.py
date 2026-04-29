import time
import torch
import torch.nn as nn
import joblib
import numpy as np
import os
import warnings
from dotenv import load_dotenv
from utils import sampler, scale_supervised, scale_unsupervised,sample_by_index

warnings.filterwarnings("ignore")
load_dotenv()

# Thresholds
AE_THRESHOLD = 0.001068
ISO_THRESHOLD = 0.233762


DROPOUT = 0.2

class FFNN(nn.Module):
    def __init__(self, in_dim=34, hidden1=128, hidden2=64, num_classes=7, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class Autoencoder(nn.Module):
    def __init__(self, in_dim=69, hidden1=128, hidden2=64, bottleneck=16, dropout=DROPOUT):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, in_dim),
            nn.Sigmoid(),
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# Pre-load Models
DEVICE = torch.device("cpu")

# Multiclass Models
FFNN_MODEL = FFNN(in_dim=34)
FFNN_MODEL.load_state_dict(torch.load("models/ffnn_multiclass.pt", map_location=DEVICE))
FFNN_MODEL.eval()

LIGHTGBM_MODEL = joblib.load(os.getenv('PATH_LIGHTGBM'))

# Binary / Anomaly Models
LOGREG_W = np.load("models/logreg_W.npy")
LOGREG_B = np.load("models/logreg_b.npy")

ISO_FOREST = joblib.load("models/isolation_forest.joblib")

AE_MODEL = Autoencoder(in_dim=69)
AE_MODEL.load_state_dict(torch.load("models/autoencoder_model.pth", map_location=DEVICE))
AE_MODEL.eval()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Load XGBoost Model
XGBOOST_MODEL = joblib.load(os.getenv('PATH_XGBOOST'))

# Prediction Functions

def _predict_ffnn(raw_sample, scaled_data=None):
    start_time = time.perf_counter()
    if scaled_data is None:
        scaled_data = scale_supervised(raw_sample)
    X_tensor = torch.tensor(scaled_data, dtype=torch.float32)
    with torch.no_grad():
        logits = FFNN_MODEL(X_tensor)
        preds = logits.argmax(dim=1)
    latency = time.perf_counter() - start_time
    # Use .item() to safely get the scalar value
    return int(preds.item()), latency

def _predict_lightgbm(raw_sample, scaled_data=None):
    start_time = time.perf_counter()
    if scaled_data is None:
        scaled_data = scale_supervised(raw_sample)
    y_pred = LIGHTGBM_MODEL.predict(scaled_data)
    latency = time.perf_counter() - start_time
    # Use .ravel()[0] to handle both scalars and arrays
    return int(np.atleast_1d(y_pred).ravel()[0]), latency

def _predict_xgboost(raw_sample, scaled_data=None):
    start_time = time.perf_counter()
    if scaled_data is None:
        scaled_data = scale_supervised(raw_sample)
    # Use .item() or .ravel()[0] for safety
    y_pred = XGBOOST_MODEL.predict(scaled_data)
    latency = time.perf_counter() - start_time
    return int(np.atleast_1d(y_pred).ravel()[0]), latency

def _predict_logreg(raw_sample, scaled_data=None):
    start_time = time.perf_counter()
    if scaled_data is None:
        scaled_data = scale_supervised(raw_sample)
    
    # Manually compute linear layer (W * x + b)
    logits = np.dot(scaled_data, LOGREG_W.T) + LOGREG_B
    # Use .item() or .ravel()[0] for safety
    val = np.atleast_1d(logits).ravel()[0]
    prediction = 1 if val > 0 else 0
    latency = time.perf_counter() - start_time
    return prediction, latency

def _predict_isolation_forest(raw_sample, scaled_data=None):
    start_time = time.perf_counter()
    if scaled_data is None:
        scaled_data = scale_unsupervised(raw_sample)
    score = ISO_FOREST.decision_function(scaled_data)
    prediction = 1 if score[0] < ISO_THRESHOLD else 0
    latency = time.perf_counter() - start_time
    return prediction, latency

def _predict_autoencoder(raw_sample, scaled_data=None):
    start_time = time.perf_counter()
    if scaled_data is None:
        scaled_data = scale_unsupervised(raw_sample)
    X_tensor = torch.tensor(scaled_data, dtype=torch.float32)
    with torch.no_grad():
        reconstructed = AE_MODEL(X_tensor)
        sq_errors = (X_tensor - reconstructed) ** 2
        mse = torch.mean(sq_errors, dim=1)
        max_err = torch.max(sq_errors, dim=1).values
        combined_error = 0.5 * (mse + max_err)
        error_val = combined_error.item()
    
    # 1 if combined error is above threshold, else 0
    prediction = 1 if error_val > AE_THRESHOLD else 0
    latency = time.perf_counter() - start_time
    return prediction, latency

# # --- Legacy Public Functions (For compatibility if needed) ---

# def predict_ffnn(target_class):
#     raw_sample, idx = sampler(target_class)
#     return [_predict_ffnn(raw_sample)], idx

# def predict_lightgbm(target_class):
#     raw_sample, idx = sampler(target_class)
#     return [_predict_lightgbm(raw_sample)], idx

# def predict_logreg(target_class):
#     raw_sample, idx = sampler(target_class)
#     return [_predict_logreg(raw_sample)], idx

# def predict_isolation_forest(target_class):
#     raw_sample, idx = sampler(target_class)
#     return [_predict_isolation_forest(raw_sample)], idx

# def predict_autoencoder(target_class):
#     raw_sample, idx = sampler(target_class)
#     return [_predict_autoencoder(raw_sample)], idx

# New Pipeline Function

def predict_attack(target_class):
    raw_sample, idx = sampler(target_class)
    
    total_start = time.perf_counter()
    
    # Pre-scale once for unsupervised models
    scaled_unsupervised = scale_unsupervised(raw_sample)
    
    # Unsupervised Check
    ae_flag, ae_lat = _predict_autoencoder(raw_sample, scaled_data=scaled_unsupervised)
    iso_flag, iso_lat = _predict_isolation_forest(raw_sample, scaled_data=scaled_unsupervised)
    
    result = {
        "target_class" : int(target_class),
        "status": "Benign",
        "row_index": int(idx),
        "unsupervised": {
            "autoencoder": ae_flag,
            "isolation_forest": iso_flag
        },
        "latencies": {
            "autoencoder": round(ae_lat, 6),
            "isolation_forest": round(iso_lat, 6)
        }
    }
    
    # Supervised Check 
    if ae_flag == 1 or iso_flag == 1:
        # Pre-scale once for all supervised models
        scaled_supervised = scale_supervised(raw_sample)
        
        # Final decision rests with LightGBM
        lgbm_pred, lgbm_lat = _predict_lightgbm(raw_sample, scaled_data=scaled_supervised)
        ffnn_pred, ffnn_lat = _predict_ffnn(raw_sample, scaled_data=scaled_supervised)
        logreg_pred, logreg_lat = _predict_logreg(raw_sample, scaled_data=scaled_supervised)
        xgb_pred, xgb_lat = _predict_xgboost(raw_sample, scaled_data=scaled_supervised)
        
        if lgbm_pred != 0:
            result["status"] = "Attack Detected"
            
        result["supervised"] = {
            "ffnn": ffnn_pred,
            "lightgbm": lgbm_pred,
            "logreg": logreg_pred,
            "xgboost": xgb_pred
        }
        result["latencies"].update({
            "ffnn": round(ffnn_lat, 6),
            "lightgbm": round(lgbm_lat, 6),
            "logreg": round(logreg_lat, 6),
            "xgboost": round(xgb_lat, 6)
        })
    
    result["total_detection_time"] = round(time.perf_counter() - total_start, 6)
    return result


def predict_attack_by_idx(idx):
    raw_sample, target_class = sample_by_index(idx)
    
    total_start = time.perf_counter()
    
    # Pre-scale once for unsupervised models
    scaled_unsupervised = scale_unsupervised(raw_sample)
    
    # Unsupervised Check
    ae_flag, ae_lat = _predict_autoencoder(raw_sample, scaled_data=scaled_unsupervised)
    iso_flag, iso_lat = _predict_isolation_forest(raw_sample, scaled_data=scaled_unsupervised)
    
    result = {
        "target_class" : int(target_class),
        "status": "Benign",
        "row_index": int(idx),
        "unsupervised": {
            "autoencoder": ae_flag,
            "isolation_forest": iso_flag
        },
        "latencies": {
            "autoencoder": round(ae_lat, 6),
            "isolation_forest": round(iso_lat, 6)
        }
    }
    
    # Supervised Check 
    if ae_flag == 1 or iso_flag == 1:
        # Pre-scale once for all supervised models
        scaled_supervised = scale_supervised(raw_sample)
        
        # Final decision rests with LightGBM
        lgbm_pred, lgbm_lat = _predict_lightgbm(raw_sample, scaled_data=scaled_supervised)
        ffnn_pred, ffnn_lat = _predict_ffnn(raw_sample, scaled_data=scaled_supervised)
        logreg_pred, logreg_lat = _predict_logreg(raw_sample, scaled_data=scaled_supervised)
        xgb_pred, xgb_lat = _predict_xgboost(raw_sample, scaled_data=scaled_supervised)
        
        if lgbm_pred != 0:
            result["status"] = "Attack Detected"
            
        result["supervised"] = {
            "ffnn": ffnn_pred,
            "lightgbm": lgbm_pred,
            "logreg": logreg_pred,
            "xgboost": xgb_pred
        }
        result["latencies"].update({
            "ffnn": round(ffnn_lat, 6),
            "lightgbm": round(lgbm_lat, 6),
            "logreg": round(logreg_lat, 6),
            "xgboost": round(xgb_lat, 6)
        })
    
    result["total_detection_time"] = round(time.perf_counter() - total_start, 6)
    return result

if __name__ == "__main__":
    import json
    print("Testing Pipeline (Class 0):")
    print(json.dumps(predict_attack(0), indent=2))
    print("\nTesting Pipeline (Class 4):")
    print(json.dumps(predict_attack(4), indent=2))
    # # print("Testing Pipeline (Class 0):")
    # # print(json.dumps(predict_attack(0), indent=2))
    # print("\nTesting Pipeline (idx  17502 ):")
    # print(json.dumps(predict_attack_by_idx(17502), indent=2))