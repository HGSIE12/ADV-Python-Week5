import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold

# 1) Load dataset
df = pd.read_csv("data_day24_selection.csv")

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap - BEFORE")
plt.show()


selector = VarianceThreshold(threshold=0.9)
X_var = selector.fit_transform(df)

kept_columns = df.columns[selector.get_support()]
df_var = pd.DataFrame(X_var, columns=kept_columns)

print("Columns kept after VarianceThreshold:")
print(kept_columns)


def drop_correlated_features(df, threshold=0.9):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=to_drop), to_drop


df_final, dropped_cols = drop_correlated_features(df_var, threshold=0.9)

print("\nDropped due to high correlation:")
print(dropped_cols)


plt.figure(figsize=(8,6))
sns.heatmap(df_final.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap - AFTER")
plt.show()
