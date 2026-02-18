import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# 1) Load regression dataset
df = pd.read_csv("day23_poly.csv")

X = df[['x']].values
y = df['y'].values

#2) Add polynomial features (degrees 1,2,5)
for d in [1, 2, 5]:
    poly = PolynomialFeatures(degree=d, include_bias=False)
    X_poly = poly.fit_transform(X)
    scores = cross_val_score(LinearRegression(), X_poly, y, cv=5)

    #3) Compare model fits or visualize predictions
    print(f"Degree {d}, CV R²: {scores.mean():.3f} (std {scores.std():.3f})")

