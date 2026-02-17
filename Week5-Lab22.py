import pandas as pd

#1) Load interaction dataset

df = pd.read_csv("day22_interactions.csv")
print(df)

#2) Create multiplicative, additive, and logical interactions

df["feature3"] = df["feature1"] * df["feature2"]
print(df)

df["target2"] = df["feature3"] + df["feature2"]
print(df)
df["high_risk"] = ((df["feature3"] > 6) & (df["target"] > 4)).astype(int)
print(df)

#3) Compute correlations with target

corr_matrix = df[["feature3", "feature1", "feature2"]].corr()
print(corr_matrix["feature1"])
