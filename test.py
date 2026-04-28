import polars as pd

df = pd.read_parquet("unscaled_test.parquet")

print(df.shape[0])