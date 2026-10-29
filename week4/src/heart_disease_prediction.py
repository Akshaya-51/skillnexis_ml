"""
Week 4 Capstone - Heart Disease Prediction
Full pipeline: cleaning -> EDA -> scaling -> Logistic Regression / Decision Tree
-> evaluation (accuracy, precision, recall, confusion matrix, ROC) -> save model.

Run from the project root:  python src/heart_disease_prediction.py
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve, classification_report)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

ROOT = Path(__file__).resolve().parent.parent
IMG, MODELS, RESULTS = ROOT / "images", ROOT / "models", ROOT / "results"
RANDOM_STATE = 42
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------- 1. Load + clean
df = pd.read_csv(ROOT / "data" / "heart.csv")
print("Raw shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))

df = df.drop_duplicates().reset_index(drop=True)   # 1 exact duplicate row
print("Shape after cleaning:", df.shape)
print(df["target"].value_counts().rename({1: "disease", 0: "no disease"}))

# ---------------------------------------------------------------- 2. Quick EDA
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.countplot(x="target", data=df, ax=ax[0])
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["No disease (0)", "Disease (1)"])
ax[0].set_title("Class balance")
sns.heatmap(df.corr(), cmap="coolwarm", center=0, ax=ax[1], cbar_kws={"shrink": .8})
ax[1].set_title("Correlation matrix")
plt.tight_layout(); plt.savefig(IMG / "01_eda.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 3. Features + split
X, y = df.drop(columns="target"), df["target"]
categorical = ["cp", "restecg", "slope", "thal"]        # codes, not real magnitudes
numeric = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca", "sex", "fbs", "exang"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Scaling lives inside the pipeline so it is fit on training folds only (no leakage)
def make_pipeline(model, scale=True):
    pre = ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([("prep", pre), ("model", model)])

# ---------------------------------------------------------------- 4. Models
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

log_reg = make_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))

# Decision trees overfit easily on 300 rows, so I tune depth/leaf size with GridSearchCV
tree_search = GridSearchCV(
    make_pipeline(DecisionTreeClassifier(random_state=RANDOM_STATE), scale=True),
    param_grid={"model__max_depth": [2, 3, 4, 5, 6, None],
                "model__min_samples_leaf": [1, 3, 5, 10],
                "model__criterion": ["gini", "entropy"]},
    cv=cv, scoring="roc_auc", n_jobs=-1)
tree_search.fit(X_train, y_train)
print("Best tree params:", tree_search.best_params_)
tree = tree_search.best_estimator_

models = {"Logistic Regression": log_reg.fit(X_train, y_train),
          "Decision Tree (tuned)": tree}

# ---------------------------------------------------------------- 5. Evaluation
rows, probs = [], {}
for name, m in models.items():
    cv_acc = cross_val_score(m, X_train, y_train, cv=cv, scoring="accuracy")
    pred = m.predict(X_test)
    prob = m.predict_proba(X_test)[:, 1]
    probs[name] = prob
    rows.append({"Model": name,
                 "CV accuracy (train, 5-fold)": f"{cv_acc.mean():.3f} +/- {cv_acc.std():.3f}",
                 "Test accuracy": accuracy_score(y_test, pred),
                 "Precision": precision_score(y_test, pred),
                 "Recall": recall_score(y_test, pred),
                 "F1": f1_score(y_test, pred),
                 "ROC AUC": roc_auc_score(y_test, prob)})
    print(f"\n=== {name} ===")
    print(classification_report(y_test, pred, target_names=["No disease", "Disease"]))

results = pd.DataFrame(rows)
results.round(3).to_csv(RESULTS / "model_comparison.csv", index=False)
print("\n", results.round(3).to_string(index=False))

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, m) in zip(axes, models.items()):
    ConfusionMatrixDisplay(confusion_matrix(y_test, m.predict(X_test)),
                           display_labels=["No disease", "Disease"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(name); ax.grid(False)
plt.tight_layout(); plt.savefig(IMG / "02_confusion_matrices.png", dpi=150); plt.close()

# ROC curves
plt.figure(figsize=(5.5, 5))
for name, p in probs.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc_score(y_test, p):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curve - heart disease"); plt.legend(loc="lower right")
plt.tight_layout(); plt.savefig(IMG / "03_roc_curves.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 6. Interpretation
feat_names = log_reg.named_steps["prep"].get_feature_names_out()
coefs = pd.Series(log_reg.named_steps["model"].coef_[0], index=feat_names).sort_values()
top = pd.concat([coefs.head(6), coefs.tail(6)])
plt.figure(figsize=(7, 5))
top.plot.barh(color=np.where(top > 0, "#d1495b", "#2e86ab"))
plt.title("Logistic Regression - strongest coefficients\n(red raises odds of target = 1, blue lowers them)")
plt.tight_layout(); plt.savefig(IMG / "04_logreg_coefficients.png", dpi=150); plt.close()

plt.figure(figsize=(14, 6))
plot_tree(tree.named_steps["model"], feature_names=feat_names,
          class_names=["No disease", "Disease"], filled=True, rounded=True, fontsize=7)
plt.title("Tuned decision tree"); plt.tight_layout()
plt.savefig(IMG / "05_decision_tree.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 7. Save best model
best_name = results.sort_values("ROC AUC", ascending=False).iloc[0]["Model"]
joblib.dump(models[best_name], MODELS / "heart_disease_model.joblib")
print(f"\nSaved best model by ROC AUC: {best_name}")
