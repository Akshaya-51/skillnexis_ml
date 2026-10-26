"""Mini Project 1: Titanic Survival Prediction - Data Cleaning Project.
Tasks: clean missing data, encode Sex & Embarked, visualize age distribution,
output cleaned dataset as a new CSV.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("data/titanic.csv")
print("Missing values before cleaning:\n", df.isnull().sum(), "\n")

# Visualize age distribution (before imputation)
plt.figure(figsize=(8, 5))
sns.histplot(df["Age"].dropna(), bins=30, kde=True)
plt.title("Age Distribution (original, missing dropped)")
plt.xlabel("Age"); plt.tight_layout()
plt.savefig("outputs/age_distribution_before.png", dpi=150); plt.close()

# 1. Clean missing data
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = df.drop(columns=["Cabin"])  # ~77% missing -> not useful
df = df.drop(columns=["PassengerId", "Name", "Ticket"])  # identifiers, not predictive

# 2. Encode Sex and Embarked
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], dtype=int)

# 3. Visualize age distribution (after cleaning) by survival
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Age", hue="Survived", bins=30, kde=True, multiple="stack")
plt.title("Age Distribution by Survival (cleaned)")
plt.tight_layout()
plt.savefig("outputs/age_distribution_after.png", dpi=150); plt.close()

# 4. Save cleaned CSV
df.to_csv("outputs/titanic_cleaned.csv", index=False)
print("Missing values after cleaning:", int(df.isnull().sum().sum()))
print("Saved outputs/titanic_cleaned.csv", df.shape)
print(df.head())
