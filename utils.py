import polars as pl
import numpy as np
import joblib
import os
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# Pre-load models for efficiency
MINMAX = joblib.load(os.getenv('PATH_MINMAX'))
STANDARD = joblib.load(os.getenv('PATH_STANDARD'))
PCA = joblib.load(os.getenv('PATH_PCA'))

# Load dataset once into memory using Polars and add a row index
df = pl.read_parquet("unscaled_test.parquet").with_row_index("original_index")
# Separate features, target, and index
y_test = df["Attack"]
indices = df["original_index"]
x = df.drop(["Attack", "original_index"])

def sampler(target_class):
    """
    Samples one row and its original index using Polars.
    """
    # Filter for the target class and sample 1 row
    # We filter the whole df to keep the relationship between x and original_index
    sample_df = df.filter(pl.col("Attack") == target_class).sample(1)
    
    if sample_df.height == 0:
        raise ValueError(f"No data found for class {target_class}")
    
    idx = sample_df["original_index"][0]
    # Drop Attack and index columns for the model input
    features = sample_df.drop(["Attack", "original_index"]).to_numpy()
    
    return features, idx

def scale_unsupervised(flow):
    return MINMAX.transform(flow)

def scale_supervised(flow):
    flow = STANDARD.transform(flow)
    flow = PCA.transform(flow)
    return flow
