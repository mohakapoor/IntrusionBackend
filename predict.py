import torch
import joblib
import torch.nn as nn
import numpy as np
import os
import pandas as pd
import lightgbm
from dotenv import load_dotenv




df = pd.read_parquet("test_mc.parquet")
x = df.drop('Attack', axis=1)
y_test = df['Attack']





load_dotenv()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class FFNN(nn.Module):
    def __init__(self, in_dim=34, hidden1=128, hidden2=64, num_classes=7, dropout=0.2):
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


def predict_ffnn(target_class):
    
    DEVICE = torch.device("cpu")

    model = FFNN()
    state_dict = torch.load("models/ffnn_multiclass.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    
    # Sample from the global x based on target_class
    try:
        sample = x[y_test == target_class].sample(1)
    except ValueError:
        raise ValueError(f"No data found for class {target_class}")
        
    x_np = sample.values.astype(np.float32)
    X_tensor = torch.tensor(x_np, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_tensor)
        preds = logits.argmax(dim=1)
    preds = preds.numpy()
    return preds



def predict_lightgbm(target_class):
    lgbm = joblib.load(os.getenv('PATH_LIGHTGBM'))
    
    # Sample from the global x based on target_class
    try:
        sample = x[y_test == target_class].sample(1)
    except ValueError:
        raise ValueError(f"No data found for class {target_class}")

    y = lgbm.predict(sample)

    
    return y



def predict_logreg(target_class):
    W = np.load("models/logreg_W.npy")
    b = np.load("models/logreg_b.npy")
    
    # Sample from the global x based on target_class
    try:
        sample = x[y_test == target_class].sample(1)
    except ValueError:
        raise ValueError(f"No data found for class {target_class}")
        
    y = (sigmoid(sample@W +b)>0.5).astype(int)
    # values, counts = np.unique(y, return_counts=True)
    
    # print(values)
    # print(counts)
    return y



print(predict_logreg(4))
print(predict_lightgbm(4))
print(predict_ffnn(4))