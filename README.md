# Week 3: Unsupervised Learning & Model Evaluation

**SkillNexis Machine Learning & AI Internship**

During Week 3 of my SkillNexis Machine Learning & AI Internship, I explored unsupervised learning techniques, dimensionality reduction, feature scaling, and model saving/loading. I used the Iris dataset to apply K-Means clustering, visualize the clusters, compare them with the actual flower labels, and use Principal Component Analysis (PCA) to reduce the data to two dimensions.

## What I Have Included in This Repository

| Task | Notebook |
|---|---|
| Assignment 1: Apply K-Means to the Iris dataset and visualize clusters | `notebooks/01_kmeans_iris_clustering.ipynb` |
| Practice Q1: Apply K-Means and print cluster centers | `notebooks/01_kmeans_iris_clustering.ipynb` |
| Practice Q2: Visualize clusters using Matplotlib | `notebooks/01_kmeans_iris_clustering.ipynb` |
| Practice Q3: Reduce the Iris dataset to 2D using PCA | `notebooks/02_pca_dimensionality_reduction.ipynb` |
| Practice Q4: Find the explained variance ratio | `notebooks/02_pca_dimensionality_reduction.ipynb` |
| Mini Project 3: Iris Flower Clustering Project — K-Means with k=3, cluster visualization, and comparison with true labels | `notebooks/01_kmeans_iris_clustering.ipynb` |
| Additional practice: Elbow method and saving/loading a model with joblib and pickle | Both notebooks |

## How to Run the Notebooks

I use Python and Jupyter Notebook to run the project.

1. Extract this ZIP file.
2. Open a terminal in the extracted `Week_3_Supervised_Learning_Solutions` folder.
3. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

4. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open either notebook from the `notebooks/` folder and select **Kernel → Restart & Run All**.

The notebooks use the Iris dataset included with scikit-learn, so I do not need to download a separate dataset. They save plots in `images/` and trained models in `models/`.

## My Approach

- **Feature scaling:** I used `StandardScaler` before K-Means and PCA because the algorithms can be affected by differences in feature scales.
- **K-Means clustering:** I applied K-Means with `k=3`, printed the cluster centers in the original feature units, and visualized the cluster assignments.
- **Choosing K:** I used the elbow method to compare inertia across different values of K.
- **Cluster evaluation:** I compared the predicted clusters with the Iris species labels using a contingency table and the Adjusted Rand Index. Since cluster IDs are assigned arbitrarily, cluster number 0 does not automatically mean species 0.
- **PCA:** I reduced the four Iris features to two principal components and calculated the explained variance ratio.
- **Model persistence:** I demonstrated saving and loading the trained K-Means model with both `joblib` and `pickle`.

## Results

The notebooks calculate the results when I run them. I can copy the printed values into this table after execution.

| Experiment | Metric | Result |
|---|---|---|
| K-Means (`k=3`) | Inertia | 139.82|
| K-Means (`k=3`) | Silhouette score | 0.46 |
| K-Means vs. true species | Adjusted Rand Index | 0.620 |
| PCA with 2 components | Total explained variance | 0.9581 (95.81%) |

## Key Learnings

Through these tasks, I gained practical experience with unsupervised learning, clustering, feature scaling, dimensionality reduction, and model persistence. I also learned that K-Means groups samples based on feature similarity rather than known labels, while PCA creates a lower-dimensional representation that preserves as much variance as possible.
