import pandas as pd

#1) Load housing data

df = pd.read_csv("day21_housing.csv")
print(df)

df["price_per_sqft"] = df["price"] / df["sqft"]
print(df)

df["room"] = df["bathrooms"] + df["bedrooms"]
print(df)

df["high_sqft"] = df["sqft"].apply(
    lambda x: "high" if x > 1700 else "low"
)
print(df)

