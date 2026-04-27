import polars as pl
import numpy as np
import joblib
import os
from dotenv import load_dotenv

load_dotenv()

MINMAX = joblib.load(os.getenv('PATH_MINMAX'))
STANDARD = joblib.load(os.getenv('PATH_STANDARD'))
PCA = joblib.load(os.getenv('PATH_PCA'))

df = pl.read_parquet("unscaled_test.parquet")
y_test = df["Attack"]
x = df.drop("Attack")

def sampler(target_class):
    sample = x.filter(y_test == target_class).sample(1)
    if sample.height == 0:
        raise ValueError(f"No data found for class {target_class}")
    return sample.to_numpy()

def scale_unsupervised(flow):
    return MINMAX.transform(flow)

def scale_supervised(flow):
    flow = STANDARD.transform(flow)
    flow = PCA.transform(flow)
    return flow
