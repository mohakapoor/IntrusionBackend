import polars as pl
import numpy as np
import joblib
import os
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# Pre-load scalars 
MINMAX = joblib.load(os.getenv('PATH_MINMAX'))
STANDARD = joblib.load(os.getenv('PATH_STANDARD'))
PCA = joblib.load(os.getenv('PATH_PCA'))


# loading data
df = pl.read_parquet("unscaled_test.parquet").with_row_index("original_index")
y_test = df["Attack"]
indices = df["original_index"]
x = df.drop(["Attack", "original_index"])


def len_df():
    return df.shape[0]

def sampler(target_class):

    sample_df = df.filter(pl.col("Attack") == target_class).sample(1)
    
    if sample_df.height == 0:
        raise ValueError(f"No data found for class {target_class}")
    
    idx = sample_df["original_index"][0]
    features = sample_df.drop(["Attack", "original_index"]).to_numpy()
    
    return features, idx


def sample_by_index(idx):
    sample_df = df.filter(pl.col("original_index") == idx)

    if sample_df.height == 0:
        raise ValueError(f"No data found for index {idx}")
    
    target_class = sample_df["Attack"][0]

    features = sample_df.drop(["Attack", "original_index"]).to_numpy()

    return features,target_class

def scale_unsupervised(flow):
    return MINMAX.transform(flow)

def scale_supervised(flow):
    flow = STANDARD.transform(flow)
    flow = PCA.transform(flow)
    return flow
