# Customer Segmentation — K-Means Clustering + PCA

## Objective
Segment mall customers into groups based on demographic and spending behavior using
K-Means clustering, and use Principal Component Analysis (PCA) to visualize those
clusters in two dimensions.

## Dataset
Mall Customer Segmentation Dataset (Kaggle):
https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

> The dataset CSV is not redistributed in this repo per the assignment instructions.
> Download it from the Kaggle link above and place it in the repo root as
> `mall_customers.csv` before running the notebook.

## Libraries Used
- pandas
- numpy
- matplotlib
- scikit-learn (`StandardScaler`, `LabelEncoder`, `KMeans`, `PCA`)

## Methodology
1. **Data Understanding** — loaded the dataset (200 rows), inspected `.info()` /
   `.describe()`, and identified `Age`, `Annual Income (k$)`, and `Spending Score
   (1-100)` as numerical features and `Genre` (gender) as the categorical feature.
   There is no target variable — this is unsupervised clustering.
2. **Data Preprocessing** — confirmed no missing values, dropped the non-predictive
   `CustomerID` column, label-encoded `Genre`, and standardized all four features with
   `StandardScaler`.
3. **Model Development** — ran the elbow method (K=1 to 10) on the scaled features,
   selected K=5 where the inertia curve flattens, trained `KMeans(n_clusters=5)`,
   assigned cluster labels to each customer, and applied `PCA(n_components=2)` to the
   scaled features for visualization.
4. **Visualization & Evaluation** — plotted the elbow curve, a cluster scatter plot
   (Annual Income vs. Spending Score), and a PCA-based 2D scatter plot colored by
   cluster, then summarized each cluster's average age, income, and spending score.

## Results
- **Optimal K:** 5 (via elbow method)
- **PCA:** 2 components capture ≈ 60% of total variance across the 4 standardized
  features

| Cluster | Avg. Age | Avg. Annual Income (k$) | Avg. Spending Score | Size |
|---|---|---|---|---|
| 0 | 32.7 | 86.5 | 82.1 | 39 |
| 1 | 36.5 | 89.5 | 18.0 | 29 |
| 2 | 49.8 | 49.2 | 40.1 | 43 |
| 3 | 24.9 | 39.7 | 61.2 | 54 |
| 4 | 55.7 | 53.7 | 36.8 | 35 |

Roughly: Cluster 0 = young, high-income, high-spending (premium target); Cluster 1 =
higher-income but low-spending; Cluster 2 & 4 = older, mid-income, moderate/lower
spending; Cluster 3 = younger, lower-income but relatively high-spending.

## Conclusion
K-Means clustering (K=5, chosen via the elbow method) segmented mall customers based on
age, gender, annual income, and spending score, and PCA compressed those four
standardized features into two components — capturing about 60% of the variance — for
visualization. The resulting segments map onto intuitive business groups, such as a
high-income, high-spending segment as a natural target for premium offers, versus a
high-income, low-spending segment worth investigating separately since their income
doesn't currently translate into spend. Segmentation like this supports targeted
marketing campaigns, personalized recommendations, and more efficient allocation of
marketing budget rather than treating all customers identically. A key limitation of
K-Means is that it requires the number of clusters K to be chosen in advance and assumes
roughly spherical, similarly sized clusters, which can misrepresent naturally irregular
or overlapping customer groups. A key advantage of PCA is that it makes clusters in
higher-dimensional feature spaces visually inspectable in 2D, revealing structure that
would otherwise be impossible to see directly once more than two or three features are
involved.

## Files
- `Assignment-7.ipynb` — full notebook (data understanding, preprocessing, elbow
  method, K-Means, PCA, visualizations, and conclusion)
- `README.md` — this file
