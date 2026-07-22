# Medical Insurance Cost Prediction — Multiple Linear Regression

## Objective
Build a Multiple Linear Regression model to predict a customer's medical insurance
charges from their personal and health-related information (age, sex, BMI, number of
children, smoking status, and region).

## Dataset
Medical Cost Personal Insurance Dataset (Kaggle):
https://www.kaggle.com/datasets/mirichoi0218/insurance

> The dataset CSV is not redistributed in this repo per the assignment instructions.
> Download `insurance.csv` from the Kaggle link above and place it in the repo root
> before running the notebook.

## Libraries Used
- pandas
- numpy
- matplotlib
- scikit-learn (`train_test_split`, `LinearRegression`, evaluation metrics)

## Methodology
1. **Data Understanding** — loaded the dataset, inspected the first records, and
   identified numerical features (`age`, `bmi`, `children`), categorical features
   (`sex`, `smoker`, `region`), and the target variable (`charges`).
2. **Data Preprocessing** — checked for missing values (none found), one-hot encoded
   the categorical variables (`sex`, `smoker`, `region`), and split the data into an
   80% training set and a 20% test set (`random_state=42`).
3. **Model Development** — trained a `LinearRegression` model on the encoded training
   data using all six original features, then generated predictions on the test set.
4. **Model Evaluation** — evaluated the model with MAE, MSE, RMSE, and R², and plotted
   actual vs. predicted charges for the test set.

## Results
| Metric | Value |
|---|---|
| MAE | ≈ 4,181 |
| MSE | ≈ 33,596,916 |
| RMSE | ≈ 5,796 |
| R² Score | ≈ 0.78 |

Smoking status was by far the strongest predictor of charges, followed by age, BMI, and
number of children; region and sex had comparatively small effects.

## Conclusion
The model explains roughly 78% of the variance in insurance charges. Smoking status
dominates the prediction, consistent with its well-known link to higher medical risk and
cost, while age, BMI, and children add smaller, positive contributions. The main
limitation of Linear Regression here is its purely additive, linear structure — it cannot
capture interaction effects (e.g., smoking combined with high BMI compounding risk) or the
right-skewed nature of real charge distributions, which likely explains the under-prediction
of the highest-cost cases visible in the actual-vs-predicted plot. Techniques such as
interaction terms, polynomial features, or tree-based ensemble models could improve on
this baseline.

## Files
- `Assignment-1.ipynb` — full notebook (data understanding, preprocessing, model,
  evaluation, plot, and conclusion)
- `README.md` — this file
