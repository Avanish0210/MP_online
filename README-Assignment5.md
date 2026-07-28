# Employee Attrition Prediction — Decision Tree vs. Random Forest

## Objective
Predict whether an employee is likely to leave the company (attrition) from
demographic, professional, and work-related attributes, and compare a Decision Tree
classifier against a Random Forest classifier (100 estimators).

## Dataset
IBM HR Analytics Employee Attrition & Performance Dataset (Kaggle):
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

> The dataset CSV is not redistributed in this repo per the assignment instructions.
> Download it from the Kaggle link above and place it in the repo root as
> `hr_attrition.csv` before running the notebook.

## Libraries Used
- pandas
- numpy
- matplotlib
- scikit-learn (`train_test_split`, `LabelEncoder`, `DecisionTreeClassifier`,
  `RandomForestClassifier`, evaluation metrics, `ConfusionMatrixDisplay`)

## Methodology
1. **Data Understanding** — loaded the dataset (1,470 rows, 35 columns), inspected
   `.info()`/`.describe()`, and identified the numerical features (Age, MonthlyIncome,
   DistanceFromHome, etc.), categorical features (BusinessTravel, Department, JobRole,
   OverTime, etc.), and `Attrition` as the target.
2. **Data Preprocessing** — confirmed no missing values, dropped `EmployeeCount`,
   `StandardHours`, and `Over18` (constant across all 1,470 rows — zero information) and
   `EmployeeNumber` (unique identifier), label-encoded the target (No → 0, Yes → 1),
   one-hot encoded the remaining categorical features, and split the data 80/20
   (stratified, since attrition is imbalanced at ~16%).
3. **Model Development** — trained a `DecisionTreeClassifier` and a
   `RandomForestClassifier` (`n_estimators=100`) on the identical training set and
   generated predictions for both on the test set.
4. **Model Evaluation & Comparison** — evaluated both models with Accuracy, Precision,
   Recall, and F1-score, plotted a confusion matrix for each side by side, and plotted
   the Random Forest's top-15 feature importances.
5. **Bonus (not for marks)** — tuned Random Forest's `max_depth` across
   {5, 10, 15, None} and reported the effect on accuracy/F1.

## Results
| Metric | Decision Tree | Random Forest |
|---|---|---|
| Accuracy | ≈ 0.765 | ≈ 0.833 |
| Precision | ≈ 0.310 | ≈ 0.417 |
| Recall | ≈ 0.383 | ≈ 0.106 |
| F1-Score | ≈ 0.343 | ≈ 0.169 |

Top Random Forest feature importances: `MonthlyIncome`, `Age`, `TotalWorkingYears`,
`DailyRate`, `HourlyRate`, `DistanceFromHome`, `OverTime`.

**Bonus tuning:** limiting Random Forest's `max_depth` to 10 gave a small improvement
over the unrestricted default (accuracy ≈0.837 vs ≈0.833, F1 ≈0.172 vs ≈0.169) —
constraining depth slightly reduces overfitting across the 100 trees, though the gain is
marginal since class imbalance (not depth) is the main factor limiting recall here.

## Model Comparison
Random Forest is more accurate and more precise overall, but the Decision Tree catches
substantially more true attrition cases (recall ≈0.38 vs ≈0.11) — Random Forest's
default behavior here is conservative, predicting "Yes" (attrition) rarely, which
inflates accuracy on this imbalanced dataset (only ~16% attrition) while missing most of
the employees who actually left. Which model is "better" depends on the business goal:
if minimizing false alarms matters most, Random Forest wins; if catching as many at-risk
employees as possible matters most, the plain Decision Tree performs better here despite
its lower headline accuracy.

## Conclusion
Both a Decision Tree and a Random Forest (100 estimators) were trained to predict
employee attrition from 30 demographic, professional, and work-related features after
dropping constant/identifier columns and one-hot encoding the categorical variables.
Random Forest achieved the higher accuracy (≈0.83 vs ≈0.77) and precision, but the
Decision Tree actually achieved noticeably higher recall on the attrition class (≈0.38 vs
≈0.11), meaning it caught more of the employees who genuinely left, at the cost of more
false alarms — a reminder that "better model" depends on which error type the business
cares about more. Random Forest generally outperforms a single Decision Tree because it
trains many trees on bootstrapped samples with random feature subsets and averages their
votes, which reduces the variance and overfitting that a single deep tree is prone to,
usually producing more stable predictions on unseen data. A key limitation of Decision
Trees is that a single tree can overfit the training data very easily, memorizing noise
specific to the training rows, which hurts generalization. A key limitation of Random
Forest is that it trades away most of a single tree's interpretability, and — as seen
here — an ensemble tuned for overall accuracy can end up biased toward the majority class
on an imbalanced target unless that imbalance is explicitly addressed (e.g., class
weighting, resampling, or threshold tuning).

## Files
- `Assignment-5.ipynb` — full notebook (data understanding, preprocessing, both models,
  evaluation, confusion matrices, feature importance plot, bonus tuning, and conclusion)
- `README.md` — this file
