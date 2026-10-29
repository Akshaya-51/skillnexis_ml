"""
Week 4 Practice Set - Cross-validation, GridSearchCV, Model Saving
Dataset: the same heart disease CSV I use in the capstone.

Run from the project root:  python practice_set/practice_solutions.py
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
RANDOM_STATE = 42

df = pd.read_csv(ROOT / "data" / "heart.csv").drop_duplicates()
X, y = df.drop(columns="target"), df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

# ---------------------------------------------------------------------------
# Q1. Use cross_val_score to test accuracy
# SVMs are scale-sensitive, so scaling goes inside the pipeline (fit per fold).
# ---------------------------------------------------------------------------
print("Q1: cross_val_score")
base_svm = make_pipeline(StandardScaler(), SVC(random_state=RANDOM_STATE))
scores = cross_val_score(base_svm, X_train, y_train, cv=5, scoring="accuracy")
print("  fold accuracies:", scores.round(3))
print(f"  mean accuracy: {scores.mean():.3f}  (std {scores.std():.3f})\n")

# ---------------------------------------------------------------------------
# Q2. Implement GridSearchCV for SVM
# ---------------------------------------------------------------------------
print("Q2: GridSearchCV for SVM")
param_grid = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__kernel": ["linear", "rbf", "poly"],
    "svc__gamma": ["scale", 0.1, 0.01, 0.001],
}
grid = GridSearchCV(make_pipeline(StandardScaler(), SVC(random_state=RANDOM_STATE)),
                    param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)
print("  best params:", grid.best_params_)
print(f"  best CV accuracy: {grid.best_score_:.3f}")
print(f"  test accuracy of best model: {grid.score(X_test, y_test):.3f}\n")

# ---------------------------------------------------------------------------
# Q3. Save the trained model using joblib
# ---------------------------------------------------------------------------
model_path = ROOT / "models" / "svm_best_model.joblib"
joblib.dump(grid.best_estimator_, model_path)
print(f"Q3: model saved to {model_path.relative_to(ROOT)}\n")

# ---------------------------------------------------------------------------
# Q4. Load the model and predict on new data
# The "new patients" are made-up rows in the same column order as the dataset.
# ---------------------------------------------------------------------------
loaded = joblib.load(model_path)
new_patients = pd.DataFrame([
    # age sex cp trestbps chol fbs restecg thalach exang oldpeak slope ca thal
    [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1],   # first row of the dataset (label = 1), as a sanity check
    [41, 0, 1, 130, 204, 0, 0, 172, 0, 1.4, 2, 0, 2],   # younger female, high max heart rate
    [57, 1, 0, 140, 192, 0, 1, 148, 0, 0.4, 1, 0, 1],
], columns=X.columns)
preds = loaded.predict(new_patients)
print("Q4: predictions on new data")
for i, p in enumerate(preds, 1):
    print(f"  patient {i}: {'heart disease likely' if p == 1 else 'no heart disease likely'}")
