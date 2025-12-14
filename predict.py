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


def predict_ffnn(x):
    DEVICE = torch.device("cpu")

    model = FFNN()
    state_dict = torch.load("models/ffnn_multiclass.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    x_np = x.values.astype(np.float32)
    X_tensor = torch.tensor(x_np, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_tensor)
        preds = logits.argmax(dim=1)
    preds = preds.numpy()
    return preds



def predict_lightgbm(x):
    lgbm = joblib.load(os.getenv('PATH_LIGHTGBM'))
    y = lgbm.predict(x)

    
    return y



def predict_logreg(x):
    W = np.load("models/logreg_W.npy")
    b = np.load("models/logreg_b.npy")
    y = (sigmoid(x@W +b)>0.5).astype(int)
    # values, counts = np.unique(y, return_counts=True)
    
    # print(values)
    # print(counts)
    return y



print(predict_logreg(x.iloc[[0]]))
print(predict_lightgbm(x.iloc[[0]]))
print(predict_ffnn(x.iloc[[0]]))