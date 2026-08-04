"""Small smoke test for the Flask health and prediction endpoints."""
from app import app

SAMPLE = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
}


def main():
    with app.test_client() as client:
        health = client.get("/health")
        prediction = client.post("/predict", json=SAMPLE)
        assert health.status_code == 200
        assert prediction.status_code == 200
        assert prediction.json["prediction"] in {
            "Heart Disease Detected", "No Heart Disease Detected"
        }
        print("API smoke test passed:", prediction.json)


if __name__ == "__main__":
    main()
