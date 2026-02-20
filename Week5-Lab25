import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression

# 1) Load regression dataset
df = pd.read_csv("day25_project.csv")
print("Original Data")
print(df)

# =====================================
# 2) Apply domain features
# =====================================
df["price_per_sqft"] = df["price"] / df["sqft"]
print("\n", df)

# =====================================
# 2) Apply interactions features
# =====================================
df["price_per_room"] = df["price"] / df["rooms"]
print("\n", df)

# =====================================
# 3) transformations
# =====================================
y = df["price"]

X = df[["sqft", "rooms"]]

poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X)

print("Original shape:", X.shape)
print("Expanded shape:", X_poly.shape)

feature_names = poly.get_feature_names_out(X.columns)
df_poly = pd.DataFrame(X_poly, columns=feature_names)

# =====================================
# 4) Feature Selection
# =====================================
selector = SelectKBest(score_func=f_regression, k=3)
X_selected = selector.fit_transform(df_poly, y)

selected_features = df_poly.columns[selector.get_support()]
df_selected = pd.DataFrame(X_selected, columns=selected_features)

print("\nSelected features:")
print(selected_features)

# =====================================
# 5) Save engineered dataset
# =====================================
final_df = pd.concat([df_selected, y.reset_index(drop=True)], axis=1)

final_df.to_csv("day25_project_engineered.csv", index=False)
