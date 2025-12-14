import pandas as pd
df = pd.read_parquet("test_mc.parquet")
x = df.drop('Attack', axis=1)
y_test = df['Attack']


print(x[y_test == 4].iloc[0].tolist()
)