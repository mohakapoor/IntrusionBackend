import os
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# Import project modules
import predict
from utils import scale_supervised, scale_unsupervised

def evaluate_cross_dataset(dataset_path, output_dir="cross_dataset_results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Loading cross-dataset validation data from {dataset_path}...")
    df = pl.read_parquet(dataset_path)
    
    y_true = df["Attack"].to_numpy()
    features_df = df.drop(["Attack"])
    
    # Identify multi-class and binary targets
    # For binary models, we treat any non-zero class as 1 (Attack)
    y_true_binary = (y_true != 0).astype(int)
    
    total_rows = len(df)
    print(f"Starting evaluation on {total_rows} rows...")
    
    # Storage for predictions and raw scores
    preds = {
        "autoencoder": [],
        "isolation_forest": [],
        "logreg": [],
        "lightgbm": [],
        "xgboost": [],
        "ffnn": [],
        "pipeline": []
    }
    
    scores = {
        "autoencoder": [],
        "isolation_forest": []
    }
    
    X_numpy = features_df.to_numpy()
    
    # Pre-load thresholds from predict.py for raw score calculation
    AE_THRESHOLD = predict.AE_THRESHOLD
    ISO_THRESHOLD = predict.ISO_THRESHOLD
    
    print("Running inference...")
    for i in tqdm(range(total_rows)):
        raw_sample = X_numpy[i].reshape(1, -1)
        
        # 1. Unsupervised Models (Getting Raw Scores for ROC AUC)
        # Replicating predict.py logic to get scores
        scaled_unsupervised = predict.scale_unsupervised(raw_sample)
        
        # Autoencoder Score
        X_tensor = predict.torch.tensor(scaled_unsupervised, dtype=predict.torch.float32)
        with predict.torch.no_grad():
            reconstructed = predict.AE_MODEL(X_tensor)
            sq_errors = (X_tensor - reconstructed) ** 2
            mse = predict.torch.mean(sq_errors, dim=1)
            max_err = predict.torch.max(sq_errors, dim=1).values
            ae_score = 0.5 * (mse + max_err).item()
        ae_flag = 1 if ae_score > AE_THRESHOLD else 0
        
        # Isolation Forest Score
        iso_score = predict.ISO_FOREST.decision_function(scaled_unsupervised)[0]
        # IF decision_function: lower is more anomalous. For ROC AUC we want higher = more anomalous
        # so we use -score
        iso_auc_score = -iso_score 
        iso_flag = 1 if iso_score < ISO_THRESHOLD else 0
        
        # 2. Supervised Models
        lgbm_pred, _ = predict._predict_lightgbm(raw_sample)
        xgb_pred, _ = predict._predict_xgboost(raw_sample)
        ffnn_pred, _ = predict._predict_ffnn(raw_sample)
        logreg_pred, _ = predict._predict_logreg(raw_sample)
        
        # 3. Hybrid Pipeline Logic
        pipeline_pred = lgbm_pred if (ae_flag == 1 or iso_flag == 1) else 0
            
        # Store predictions
        preds["autoencoder"].append(ae_flag)
        preds["isolation_forest"].append(iso_flag)
        preds["logreg"].append(logreg_pred)
        preds["lightgbm"].append(lgbm_pred)
        preds["xgboost"].append(xgb_pred)
        preds["ffnn"].append(ffnn_pred)
        preds["pipeline"].append(pipeline_pred)
        
        # Store scores for AUC
        scores["autoencoder"].append(ae_score)
        scores["isolation_forest"].append(iso_auc_score)

    # Define model categories for reporting
    multiclass_models = ["lightgbm", "xgboost", "ffnn", "pipeline"]
    binary_models = ["autoencoder", "isolation_forest", "logreg"]
    
    # CICIDS2018 subset labels
    PRESENT_LABELS = [0, 1, 2, 3, 4]
    from sklearn.metrics import roc_auc_score

    # Generate Reports
    for model_name, y_pred in preds.items():
        print(f"\nGenerating report for: {model_name}")
        
        target = y_true if model_name in multiclass_models else y_true_binary
        labels = PRESENT_LABELS if model_name in multiclass_models else [0, 1]
        
        report = classification_report(target, y_pred, labels=labels)
        cm = confusion_matrix(target, y_pred, labels=labels)
        
        # Save text report
        report_file = os.path.join(output_dir, f"{model_name}_report.txt")
        with open(report_file, "w") as f:
            f.write(f"=== {model_name.upper()} CROSS-DATASET VALIDATION ===\n")
            if model_name in scores:
                auc = roc_auc_score(y_true_binary, scores[model_name])
                f.write(f"ROC AUC SCORE: {auc:.4f}\n\n")
            f.write(report)
            f.write("\n\n=== CONFUSION MATRIX ===\n")
            f.write(str(cm))
            
        # Save Confusion Matrix Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        title = f"{model_name.replace('_', ' ').title()} CM"
        if model_name in scores:
            title += f" (AUC: {auc:.3f})"
        plt.title(f"{title}\n(CICIDS2018 Cross-Validation)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(os.path.join(output_dir, f"{model_name}_cm.png"), dpi=300)
        plt.close()

    print(f"\nAll cross-dataset validation results saved to '{output_dir}/'")

if __name__ == "__main__":
    DATASET = "cross_dataset_validation/unscaled_cicds2018.parquet"
    evaluate_cross_dataset(DATASET)
