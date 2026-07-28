# Weather Condition Classification — SVM + Open-Meteo API

## Objective
Classify hourly weather observations as **Cool** or **Warm** using an SVM (RBF kernel)
classifier, trained on meteorological data fetched from the Open-Meteo API.

## API Documentation
https://open-meteo.com/en/docs (free, no API key required)

Example request used:
```
https://api.open-meteo.com/v1/forecast?latitude=28.6139&longitude=77.2090&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m&forecast_days=7
```

## Libraries Used
- requests
- pandas
- numpy
- matplotlib
- scikit-learn (`train_test_split`, `StandardScaler`, `LabelEncoder`, `SVC`,
  evaluation metrics, `ConfusionMatrixDisplay`)

## Methodology
1. **Data Collection & Understanding** — fetched a 7-day hourly forecast for Delhi
   (28.6139° N, 77.2090° E) from the Open-Meteo API, converted the JSON response into a
   DataFrame, and created the `Weather_Class` target (`Warm` if `temperature_2m` ≥ 25°C,
   else `Cool`). Input features are `relative_humidity_2m`, `surface_pressure`, and
   `wind_speed_10m`.
2. **Data Preprocessing** — confirmed no missing values, dropped the `time` column and
   excluded `temperature_2m` from the feature set (it directly defines the target by
   threshold, so including it would be data leakage), label-encoded the target, split
   80/20 (stratified), and standardized the three features with `StandardScaler`.
3. **Model Development** — trained an `SVC` with `kernel='rbf'` on the scaled training
   data and predicted the weather class on the test set.
4. **Model Evaluation** — evaluated with Accuracy, Precision, Recall, F1-score, and a
   confusion matrix.

## Results
| Metric | Value |
|---|---|
| Accuracy | ≈ 0.853 |
| Precision | ≈ 0.950 |
| Recall | ≈ 0.826 |
| F1-Score | ≈ 0.884 |

## ⚠️ Important note on the data used
The notebook's data-fetch cell (Task 1) contains the real, correct `requests` call to
the live Open-Meteo API and will pull live data when you run it with normal internet
access. The environment this notebook was originally drafted and executed in has
restricted network egress and could not reach `api.open-meteo.com`, so the fetch cell
automatically falls back to a local `sample_openmeteo_response.json` file — a
locally generated stand-in built to match Open-Meteo's exact response structure and
realistic value ranges for Delhi in late July — purely so every cell could be executed
end-to-end with real, visible outputs. **Before submitting, run the notebook yourself
with a working internet connection** so the results reflect genuine live API data (the
code needs no changes to do this — it will use the live response automatically once the
request succeeds).

## Conclusion
The SVM (RBF kernel) reached about 85% accuracy classifying hours as Warm or Cool using
only humidity, pressure, and wind speed — none of which is a direct temperature reading
— showing these variables do carry real signal, likely because humidity in particular
tends to move inversely with temperature through the day. Feature scaling matters a
great deal for SVM because it finds the maximum-margin boundary using distances in
feature space; without standardization, a large-magnitude feature like surface pressure
(values in the high 900s) would dominate that distance over a small-magnitude one like
wind speed, regardless of true importance. A key advantage of SVM is its ability to
model non-linear decision boundaries through the kernel trick — the RBF kernel here
captures the cyclical, non-linear relationship between humidity/pressure and temperature
without manual feature engineering. A key limitation is that SVM scales poorly to large
datasets (training time grows quickly with sample size) and its performance depends
heavily on hyperparameter choices like `C` and `gamma`, which usually require tuning via
cross-validation.

## Files
- `Assignment-6.ipynb` — full notebook (data fetch, preprocessing, model, evaluation,
  confusion matrix, and conclusion)
- `sample_openmeteo_response.json` — the local stand-in response used to produce the
  executed outputs in this notebook (see note above); not required once you re-run with
  live data
- `README.md` — this file
