# Week 2: Supervised Learning (Regression & Classification)

**SkillNexis Machine Learning & AI Internship**

During Week 2 of my SkillNexis Machine Learning & AI Internship, I focused on learning and implementing supervised learning techniques for regression and classification. I worked with Linear Regression, Logistic Regression, Decision Trees, K-Nearest Neighbors (KNN), and Random Forest, along with evaluation metrics such as MSE, R² score, accuracy, and confusion matrix.

## What I Have Included in This Repository

| Task | Notebook |
|---|---|
| Assignment 1: Linear Regression on a housing dataset | `notebooks/01_house_price_prediction_linear_regression.ipynb` |
| **Mini Project 2: House Price Prediction Model** — training the model, making predictions, evaluating the R² score, and plotting predicted vs. actual prices | `notebooks/01_house_price_prediction_linear_regression.ipynb` |
| Assignment 2: Logistic Regression on the Titanic dataset | `notebooks/02_titanic_logistic_regression.ipynb` |
| Practice Q1 and Q2: Logistic Regression, confusion matrix, and accuracy evaluation | `notebooks/02_titanic_logistic_regression.ipynb` |
| Practice Q3: Comparing Decision Tree and KNN accuracy | `notebooks/03_practice_set_decision_tree_knn.ipynb` |
| Practice Q4: Tuning the `max_depth` parameter of a Decision Tree | `notebooks/03_practice_set_decision_tree_knn.ipynb` |

## How to Run the Notebooks

To run my notebooks locally, I follow these steps:

1. Download the required datasets using the instructions provided in [`data/README.md`](data/README.md).
2. Install the required libraries and launch Jupyter Notebook:

   ```bash
   pip install -r requirements.txt
   cd notebooks
   jupyter notebook
   ```

3. Open the notebooks from inside the `notebooks/` directory and select **Kernel → Restart & Run All** to execute all cells in sequence.
4. Check the generated plots in the `images/` directory.

## My Approach

- **Data preprocessing:** I used scikit-learn `Pipeline` objects to handle missing values, scale numerical features, and one-hot encode categorical features. This helps prevent data leakage by fitting preprocessing steps only on the training data.
- **Train-test split:** I used an 80/20 split with `random_state=42`. For the Titanic dataset, I used stratified splitting based on the `Survived` column to maintain the class distribution.
- **Decision Tree tuning:** I used 5-fold cross-validation on the training data to select the best `max_depth` value without using the test set for tuning.
- **Model evaluation:** I evaluated regression models using MSE, RMSE, and R² score. For classification models, I used accuracy, confusion matrices, and classification reports.

## Results


| Model | Dataset | Metric | Score |
|---|---|---|---|
| Linear Regression | House Prices | R² | 0.866316|
| Logistic Regression | Titanic | Accuracy | 0.8045|
| Decision Tree (Tuned) | Titanic | Accuracy | 0.7877 |
| KNN (k=5) | Titanic | Accuracy | 0.8212 |

## Key Learnings

Through this week's tasks, I gained practical experience in building supervised machine learning models, preprocessing datasets, comparing model performance, tuning hyperparameters, and visualizing predictions. I also learned the importance of selecting appropriate evaluation metrics and avoiding data leakage during model training.
