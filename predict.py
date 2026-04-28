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

# Prediction Functions

def _predict_ffnn(raw_sample):
    scaled_data = scale_supervised(raw_sample)
    X_tensor = torch.tensor(scaled_data, dtype=torch.float32)
    with torch.no_grad():
        logits = FFNN_MODEL(X_tensor)
        preds = logits.argmax(dim=1)
    return int(preds.numpy()[0])

def _predict_lightgbm(raw_sample):
    scaled_data = scale_supervised(raw_sample)
    y = LIGHTGBM_MODEL.predict(scaled_data)
    return int(y[0])

def _predict_logreg(raw_sample):
    scaled_data = scale_supervised(raw_sample)
    y = (sigmoid(scaled_data @ LOGREG_W + LOGREG_B) > 0.5).astype(int)
    return int(y[0])

def _predict_isolation_forest(raw_sample):
    scaled_data = scale_unsupervised(raw_sample)
    score = ISO_FOREST.decision_function(scaled_data)
    prediction = 1 if score[0] < ISO_THRESHOLD else 0
    return prediction

def _predict_autoencoder(raw_sample):
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
    return prediction

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
    
    # Unsupervised Check
    ae_flag = _predict_autoencoder(raw_sample)
    iso_flag = _predict_isolation_forest(raw_sample)
    
    result = {
        "target_class" : target_class,
        "status": "Benign",
        "row_index": int(idx),
        "unsupervised": {
            "autoencoder": ae_flag,
            "isolation_forest": iso_flag
        }
    }
    
    # Supervised Check 
    if ae_flag == 1 or iso_flag == 1:
        # Final decision rests with LightGBM
        lgbm_pred = _predict_lightgbm(raw_sample)
        
        if lgbm_pred != 0:
            result["status"] = "Attack Detected"
            
        result["supervised"] = {
            "ffnn": _predict_ffnn(raw_sample),
            "lightgbm": lgbm_pred,
            "logreg": _predict_logreg(raw_sample)
        }
    
    return result


def predict_attack_by_idx(idx):
    raw_sample,target_class= sample_by_index(idx)
    
    # Unsupervised Check
    ae_flag = _predict_autoencoder(raw_sample)
    iso_flag = _predict_isolation_forest(raw_sample)
    
    result = {
        "target_class" : target_class,
        "status": "Benign",
        "row_index": int(idx),
        "unsupervised": {
            "autoencoder": ae_flag,
            "isolation_forest": iso_flag
        }
    }
    
    # Supervised Check 
    if ae_flag == 1 or iso_flag == 1:
        # Final decision rests with LightGBM
        lgbm_pred = _predict_lightgbm(raw_sample)
        
        if lgbm_pred != 0:
            result["status"] = "Attack Detected"
            
        result["supervised"] = {
            "ffnn": _predict_ffnn(raw_sample),
            "lightgbm": lgbm_pred,
            "logreg": _predict_logreg(raw_sample)
        }
    
    return result

if __name__ == "__main__":
    import json
    # print("Testing Pipeline (Class 0):")
    # print(json.dumps(predict_attack(0), indent=2))
    # print("\nTesting Pipeline (Class 4):")
    # print(json.dumps(predict_attack(4), indent=2))
    # print("Testing Pipeline (Class 0):")
    # print(json.dumps(predict_attack(0), indent=2))
    print("\nTesting Pipeline (idx  17502 ):")
    print(json.dumps(predict_attack_by_idx(17502), indent=2))