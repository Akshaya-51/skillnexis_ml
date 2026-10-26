"""Practice Set - Week 1 (ML Basics). Questions 1-4 on a house-price dataset."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Create a sample CSV (living area in sq ft, bedrooms, price in $1000s)
# first 5 rows from the CS229 notes; remaining rows are synthetic
rng = np.random.default_rng(42)
area = np.concatenate([[2104, 1600, 2400, 1416, 3000], rng.integers(800, 4000, 95)])
price = 0.15 * area + rng.normal(0, 40, 100) + 20
pd.DataFrame({"area": area, "price": price.round(1)}).to_csv("data/houses.csv", index=False)

# Q1. Load a CSV with pandas and print first 10 rows
df = pd.read_csv("data/houses.csv")
print("Q1:\n", df.head(10), "\n")

# Q2. Split into train/test
X, y = df[["area"]], df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Q2: train", X_train.shape, "test", X_test.shape, "\n")

# Q3. Train Linear Regression and check accuracy
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Q3: R2 = {r2_score(y_test, pred):.3f}, RMSE = {mean_squared_error(y_test, pred) ** 0.5:.2f}\n")

# Q4. Predict house price from area input
user_area = float(input("Q4: Enter house area (sq ft): ") or 2000)
print(f"Predicted price: ${model.predict(pd.DataFrame({'area': [user_area]}))[0] * 1000:,.0f}")
