import pytest
from fastapi.testclient import TestClient
from app import app
import os

client = TestClient(app)

# Mock API Key for testing
os.environ["API_KEY"] = "test-key"
os.environ["MLFLOW_RUN_ID"] = "test-run-id"

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "Secure API is running!"}

def test_predict_no_auth():
    response = client.post("/predict", json={})
    assert response.status_code == 403 # Forbidden due to missing auth header

def test_predict_invalid_auth():
    response = client.post(
        "/predict", 
        json={}, 
        headers={"X-API-Key": "wrong-key"}
    )
    assert response.status_code == 403

def test_predict_invalid_payload():
    response = client.post(
        "/predict", 
        json={"invalid": "field"}, 
        headers={"X-API-Key": "test-key"}
    )
    assert response.status_code == 422 # Unprocessable Entity (Pydantic validation)

def test_predict_missing_model():
    # Model will be None because MLFLOW_RUN_ID is invalid
    response = client.post(
        "/predict", 
        json={
            "Avg_min_between_sent_tnx": 0.5,
            "Avg_min_between_received_tnx": 0.5,
            "Time_Diff_between_first_and_last_Mins": 10.0,
            "Sent_tnx": 1.0,
            "Received_Tnx": 1.0,
            "Number_of_Created_Contracts": 0.0
        }, 
        headers={"X-API-Key": "test-key"}
    )
    # The app.py logic returns 500 if model is None
    assert response.status_code == 500
    assert response.json()["detail"] == "Model is not initialized."
