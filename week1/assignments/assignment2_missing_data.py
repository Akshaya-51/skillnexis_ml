"""Assignment 2: Handle missing data using mean/median imputation."""
import pandas as pd

df = pd.read_csv("data/titanic.csv")
print("Missing before:\n", df.isnull().sum(), "\n")

df["Age"] = df["Age"].fillna(df["Age"].median())            # median: robust to outliers
df["Fare"] = df["Fare"].fillna(df["Fare"].mean())           # mean imputation (demo)
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])  # categorical -> mode

print("Missing after:\n", df[["Age", "Fare", "Embarked"]].isnull().sum())
