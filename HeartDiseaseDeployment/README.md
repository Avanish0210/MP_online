# Heart Disease Prediction - End-to-End ML Deployment

A complete machine-learning deployment project that predicts whether a patient is at risk of heart disease from 13 clinical parameters. It uses a Logistic Regression pipeline, a Flask REST API, and Render-ready deployment settings.

> **Medical notice:** This is an academic demonstration only. It must not be used for diagnosis or treatment decisions.

## Live deployment

Render URL: `https://<your-render-service-name>.onrender.com`

Replace the placeholder after creating the Render service. The health endpoint is `/health`; the prediction endpoint is `/predict`.

## Dataset and preprocessing

The included `heart.csv` is the [Heart Disease Prediction Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset). The dataset contains these numerical input features:

`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, and `thal`.

The target variable is `target` (0 = no disease, 1 = disease). `train_model.py` displays the first five records and missing-value count, uses an 80/20 stratified train/test split, imputes any missing numeric values with medians, standardizes features, trains a Logistic Regression classifier, reports accuracy, and serializes the full pipeline to `model.pkl`.

## Project structure

```text
HeartDiseaseDeployment/
├── app.py                 # Flask web app and REST API
├── train_model.py         # preprocessing, training, evaluation, serialization
├── model.pkl              # saved trained ML pipeline
├── heart.csv              # assignment dataset
├── requirements.txt       # production dependencies
├── render.yaml            # Render Blueprint configuration
└── templates/index.html   # optional browser interface
```

## Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
python app.py
```

Open `http://127.0.0.1:5000` for the form, or use the API below.

## API usage

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

Example response:

```json
{
  "heart_disease_probability": 0.81,
  "prediction": "Heart Disease Detected",
  "prediction_code": 1
}
```

Invalid or incomplete requests return a clear `400` response with the required field names.

## Deploy to Render

1. Push this project to a public GitHub repository.
2. In Render, select **New +** > **Blueprint** and connect the repository.
3. Render reads `render.yaml`, installs `requirements.txt`, starts `gunicorn app:app`, and monitors `/health`.
4. After deployment, copy the public `https://...onrender.com` URL into the **Live deployment** section above and verify it with `GET /health` and `POST /predict`.

If using a regular Web Service rather than a Blueprint, set the root directory to `HeartDiseaseDeployment`, build command to `pip install -r requirements.txt`, and start command to `gunicorn app:app`.

## Conclusion (132 words)

This project demonstrates a complete machine learning deployment workflow for heart disease prediction. A Logistic Regression pipeline was trained after checking the dataset, handling possible missing values, scaling numerical variables, and splitting the data into 80% training and 20% testing samples. The measured accuracy printed by the training script provides a clear baseline for this classification task, while saving the entire preprocessing and model pipeline prevents training-serving inconsistencies. The main deployment challenges are keeping dependency versions compatible, validating JSON input, and ensuring the cloud service starts from the correct project directory. Flask exposes the trained model through a simple REST API, and Render makes it publicly accessible. This workflow shows why MLOps matters: version control, reproducible training, packaging, testing, deployment, and monitoring turn a notebook model into a usable and maintainable application.
