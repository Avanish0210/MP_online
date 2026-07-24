# Breast Cancer Classification — K-Nearest Neighbors (KNN)

## Objective
Build a K-Nearest Neighbors classifier to predict whether a breast tumor is Malignant
(M) or Benign (B), based on diagnostic measurements from digitized images of cell
nuclei.

## Dataset
Breast Cancer Wisconsin (Diagnostic) Dataset (Kaggle):
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

> The dataset CSV is not redistributed in this repo per the assignment instructions.
> Download it from the Kaggle link above and place it in the repo root as
> `breast_cancer.csv` before running the notebook.

## Libraries Used
- pandas
- numpy
- matplotlib
- scikit-learn (`train_test_split`, `StandardScaler`, `LabelEncoder`,
  `KNeighborsClassifier`, evaluation metrics, `ConfusionMatrixDisplay`)

## Methodology
1. **Data Understanding** — loaded the dataset (569 rows, 33 columns), inspected
   `.info()`/`.describe()`, and identified the 30 numerical diagnostic features
   (radius, texture, perimeter, area, smoothness, etc. — each as mean, standard error,
   and "worst" value) and `diagnosis` as the target.
2. **Data Preprocessing** — confirmed the only missing data was a fully-empty
   `Unnamed: 32` column (an artifact of a trailing comma in the source CSV), dropped
   that column along with the non-predictive `id` column, label-encoded `diagnosis`
   (B → 0, M → 1), standardized all 30 features with `StandardScaler`, and split the
   data 80/20 (stratified on diagnosis).
3. **Model Development** — trained a `KNeighborsClassifier` with K = 5 on the scaled
   training data and predicted diagnoses on the test set.
4. **Model Evaluation** — evaluated with Accuracy, Precision, Recall, and F1-score,
   generated a confusion matrix, and additionally compared scaled vs. unscaled features
   to demonstrate the effect of standardization on KNN.

## Results
| Metric | Value |
|---|---|
| Accuracy | ≈ 0.956 |
| Precision | ≈ 0.974 |
| Recall | ≈ 0.905 |
| F1-Score | ≈ 0.938 |

Confusion matrix (test set, n=114):

|  | Predicted: Benign | Predicted: Malignant |
|---|---|---|
| **Actual: Benign** | 71 | 1 |
| **Actual: Malignant** | 4 | 38 |

Accuracy on the same split **without** feature scaling drops to ≈ 0.912, showing
standardization's direct impact on this distance-based algorithm.

## Conclusion
The K=5 KNN classifier reached about 95.6% accuracy, with strong precision (0.97) and
slightly lower recall (0.90) — the model occasionally classifies a malignant tumor as
benign, the more clinically costly error type. Feature scaling proved essential: KNN
classifies points by Euclidean distance to their nearest neighbors, so without
standardization, large-range features like `area_mean` would dominate that distance
over small-range but equally informative features like `smoothness_mean`; scaling both
to the same range let every feature contribute fairly, and measurably improved accuracy
in this project. A key limitation of KNN is that it has no real training phase — it must
compute distances to every training point at prediction time — making it computationally
expensive on large datasets and less reliable in high-dimensional feature spaces, where
the notion of "nearest neighbor" becomes less meaningful (the curse of dimensionality).

## Files
- `Assignment-4.ipynb` — full notebook (data understanding, preprocessing, model,
  evaluation, confusion matrix, and conclusion)
- `README.md` — this file
