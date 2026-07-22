# Salary Prediction — Polynomial Regression

## Objective
Build a Polynomial Regression model to predict an employee's salary from their
position level, since the true relationship between the two is non-linear.

## Dataset
Position Salaries Dataset (Kaggle):
https://www.kaggle.com/datasets/akram24/position-salaries

> The dataset CSV is not redistributed in this repo per the assignment instructions.
> Download it from the Kaggle link above and place it in the repo root as
> `position_salaries.csv` before running the notebook.

## Libraries Used
- pandas
- numpy
- matplotlib
- scikit-learn (`train_test_split`, `PolynomialFeatures`, `LinearRegression`,
  evaluation metrics)

## Methodology
1. **Data Understanding** — loaded the dataset (10 rows: `Position`, `Level`, `Salary`),
   inspected `.info()` and `.describe()`, and identified `Level` as the input feature and
   `Salary` as the target (`Position` is a text label carrying the same information as
   `Level`, so it wasn't used as a model input).
2. **Data Preprocessing** — confirmed no missing values, selected `Level` (X) and
   `Salary` (y), and split into an 80% train / 20% test set. Note: with only 10 rows
   total, the test set is just 2 points, so the evaluation metrics below are
   illustrative rather than statistically robust.
3. **Model Development** — transformed `Level` into polynomial features up to degree 3
   with `PolynomialFeatures`, then fit a `LinearRegression` model on the transformed
   features (this combination is what makes it Polynomial Regression) and predicted
   salaries on the test set.
4. **Model Evaluation** — evaluated with MAE, MSE, and R², and plotted the original data
   points alongside the fitted polynomial regression curve.

## Results
| Metric | Value |
|---|---|
| MAE | ≈ 70,635 |
| MSE | ≈ 6,263,853,283 |
| R² Score | ≈ 0.876 |

The fitted cubic curve tracks the sharp acceleration in salary at senior position
levels (e.g., the jump toward C-level and CEO) far better than a straight line would,
though both test points (Level 9 and Level 2) were somewhat over-predicted.

## Conclusion
Polynomial Regression (degree 3) captured the accelerating, non-linear growth in salary
across position levels, reaching an R² of about 0.88 on this very small test set. Linear
Regression fits a single straight line and can only represent a constant rate of change,
so it would systematically misfit both the flatter low end and the steep high end of this
data. Polynomial Regression instead adds powers of the input feature (Level, Level²,
Level³) and fits a curve, while still being a linear model in its coefficients — this
lets it follow the steep jump in salary at senior levels that a straight line would badly
underestimate. That flexibility is also its main risk: with only 10 data points, pushing
the degree higher would let the curve overfit and swing wildly between the known points
rather than generalizing.

## Files
- `Assignment-3.ipynb` — full notebook (data understanding, preprocessing, model,
  evaluation, curve plot, and conclusion)
- `README.md` — this file
