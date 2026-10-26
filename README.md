# Machine Learning & AI Internship – Week 1

ML fundamentals + data preprocessing (supervised vs unsupervised learning, linear regression, cleaning, encoding, train/test split).

## Structure
| Path | Contents |
|---|---|
| `assignments/` | Assignment 1 (explore with Pandas), 2 (mean/median imputation), 3 (LabelEncoder & OneHotEncoder) |
| `mini_project_titanic/` | Mini Project 1: Titanic data cleaning, encoding, age plots, cleaned CSV |
| `practice_set/` | Practice questions: load CSV, train/test split, Linear Regression, price prediction |
| `outputs/` | Cleaned dataset + plots |
| `data/` | Titanic dataset (Kaggle) and sample house data |

## Run (from repo root)
```bash
pip install -r requirements.txt
python assignments/assignment1_explore.py
python assignments/assignment2_missing_data.py
python assignments/assignment3_encoding.py
python mini_project_titanic/titanic_cleaning.py
python practice_set/practice_set.py
```

## Mini project summary
- Age (177 missing) → median imputation; Embarked (2 missing) → mode; Cabin dropped (~77% missing)
- Sex → 0/1; Embarked → one-hot
- Output: `outputs/titanic_cleaned.csv`, age distribution plots
