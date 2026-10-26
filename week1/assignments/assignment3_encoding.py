"""Assignment 3: Encode categoricals with LabelEncoder & OneHotEncoder."""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.read_csv("data/titanic.csv")
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# LabelEncoder: Sex -> 0/1
df["Sex_encoded"] = LabelEncoder().fit_transform(df["Sex"])

# OneHotEncoder: Embarked -> one column per port
ohe = OneHotEncoder(sparse_output=False)
arr = ohe.fit_transform(df[["Embarked"]])
ohe_df = pd.DataFrame(arr, columns=ohe.get_feature_names_out(["Embarked"]), index=df.index)
df = pd.concat([df, ohe_df], axis=1)

print(df[["Sex", "Sex_encoded", "Embarked"] + list(ohe_df.columns)].head(10))
