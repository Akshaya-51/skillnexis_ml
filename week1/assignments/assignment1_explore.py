"""Assignment 1: Load a dataset with Pandas and summarize basic stats."""
import pandas as pd

df = pd.read_csv("data/titanic.csv")
print("First 5 rows:\n", df.head(), "\n")
print("Shape:", df.shape, "\n")
df.info()
print("\nDescribe (numeric):\n", df.describe())
print("\nDescribe (categorical):\n", df.describe(exclude="number"))
print("\nMissing values per column:\n", df.isnull().sum())
