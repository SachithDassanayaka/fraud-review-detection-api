from fastapi.testclient import TestClient
from api.main import app
from unittest.mock import patch

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Fraud Review Detection API is running"}

@patch("api.main.predict")
def test_api(mock_predict):
    mock_predict.return_value = {
        "original_text": "test review",
        "cleaned_text": "test review",
        "prediction": 1,
        "predicted_label": "Fake",
        "probability": 0.9,
    }

    response = client.post("/predict", json={"text": "test review"})
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["predicted_label"] == "Fake"

@patch("api.main.predict")
def test_predict(mock_predict):
    mock_predict.return_value = {
        "original_text": "This product is wonderful and works very well",
        "cleaned_text": "this product wonderful works well",
        "prediction": 0,
        "predicted_label": "Non-Fake",
        "probability": 0.1,
    }

    response = client.post(
        "/predict", json={"text": "This product is wonderful and works very well"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "predicted_label" in data
    