from fastapi.testclient import TestClient
from api.main import app
from unittest.mock import patch

client = TestClient(app)

@patch("src.predict.predict")


def test_api(mock_predict, client):
    mock_predict.return_value = "fake"

    response = client.post("/predict", json={"text": "test review"})
    assert response.status_code == 200
    
    
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_predict():
    response = client.post(
        "/predict", json={"text": "This product is wonderful and works very well"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "predicted_label" in data
