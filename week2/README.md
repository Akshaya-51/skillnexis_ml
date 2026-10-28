# Week 2: Supervised Learning (Regression & Classification)

**SkillNexis Machine Learning & AI Internship**

Week 2 focuses on building, training and evaluating supervised models: Linear Regression, Logistic Regression, Decision Tree, KNN and Random Forest, with accuracy metrics (MSE, R², confusion matrix).

## What is in this repository

| Task | Notebook |
|---|---|
| Assignment 1: Linear Regression on a housing dataset | `notebooks/01_house_price_prediction_linear_regression.ipynb` |
| **Mini Project 2: House Price Prediction Model** (train, predict, R², predicted-vs-actual plot) | `notebooks/01_house_price_prediction_linear_regression.ipynb` |
| Assignment 2: Logistic Regression on Titanic | `notebooks/02_titanic_logistic_regression.ipynb` |
| Practice Q1 and Q2: Logistic Regression, confusion matrix, accuracy | `notebooks/02_titanic_logistic_regression.ipynb` |
| Practice Q3: Decision Tree vs KNN accuracy | `notebooks/03_practice_set_decision_tree_knn.ipynb` |
| Practice Q4: tune `max_depth` of the Decision Tree | `notebooks/03_practice_set_decision_tree_knn.ipynb` |

```
week-2/
├── notebooks/   the three Jupyter notebooks
├── data/        put the Kaggle CSVs here (see data/README.md)
├── images/      plots saved by the notebooks
├── docs/        the original week 2 brief and practice set
├── requirements.txt
└── README.md
```

## How to run

1. Download the two datasets as described in [`data/README.md`](data/README.md).
2. Install and launch:
   ```bash
   pip install -r requirements.txt
   cd notebooks
   jupyter notebook
   ```
   Run each notebook with **Kernel → Restart & Run All**. Open them from inside `notebooks/` or the data paths will not resolve. Plots are saved to `images/`.
3. Fill in the **Conclusions** cell at the end of each notebook with your own results.

## Approach

* **Preprocessing** is done inside scikit-learn `Pipeline`s (impute, scale, one-hot encode), so every transformation is fitted on training data only, which avoids data leakage.
* **Train/test split:** 80/20 with `random_state=42`; Titanic is stratified on `Survived`.
* **Decision Tree tuning** picks `max_depth` by 5-fold cross-validation on the training set, not by looking at the test set.
* **Metrics:** MSE, RMSE and R² for regression; accuracy, confusion matrix and classification report for classification.

## Results

Run the notebooks, then add your numbers here:

| Model | Dataset | Metric | Score |
|---|---|---|---|
| Linear Regression | House Prices | R² | _fill in_ |
| Logistic Regression | Titanic | Accuracy | _fill in_ |
| Decision Tree (tuned) | Titanic | Accuracy | _fill in_ |
| KNN (k=5) | Titanic | Accuracy | _fill in_ |
