import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# Import our project modules
import predict
from utils import len_df

def generate_stats():
    total_rows = len_df()
    print(f"Starting evaluation on {total_rows} rows...")
    
    y_true = []
    y_pred = []
    
    # Iterate through the dataset
    for idx in tqdm(range(total_rows)):
        try:
            result = predict.predict_attack_by_idx(idx)
            
            actual = result["target_class"]
            # Logic: If status is Attack Detected, use the LightGBM prediction. 
            # Otherwise, it's Benign (0).
            if result["status"] == "Attack Detected":
                predicted = result.get("supervised", {}).get("lightgbm", 0)
            else:
                predicted = 0
                
            y_true.append(actual)
            y_pred.append(predicted)
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            continue

    # 1. Generate Text Report
    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    output_text = "=== CONFUSION MATRIX ===\n"
    output_text += str(cm) + "\n\n"
    output_text += "=== CLASSIFICATION REPORT ===\n"
    output_text += report
    
    with open("confusion_matrix.txt", "w") as f:
        f.write(output_text)
    
    print("\nText report saved to confusion_matrix.txt")

    # 2. Generate Heatmap Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_true), 
                yticklabels=np.unique(y_true))
    plt.title("Intrusion Detection Confusion Matrix\n(Hierarchical Pipeline with LGBM Veto)")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()
    
    print("Plot saved to confusion_matrix.png")

if __name__ == "__main__":
    generate_stats()
