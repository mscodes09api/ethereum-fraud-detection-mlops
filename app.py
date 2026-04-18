import os
import secrets
import logging
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import mlflow.xgboost
import pandas as pd

# Set up logging to prevent leaking stack traces
logger = logging.getLogger(__name__)

app = FastAPI(title="Ethereum Fraud Detection API")

# --- 1. Security: CORS Middleware ---
# Pulling origins from env or defaulting to an empty list to prevent wildcards
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != [""] else ["http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# --- 2. Security: API Key Authentication ---
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

def verify_api_key(api_key: str = Security(api_key_header)):
    # FAIL-FAST: No fallback string. If API_KEY is missing from ENV, app returns 500.
    expected_key = os.environ.get("API_KEY") 
    if not expected_key:
        logger.critical("SECURITY CONFIG ERROR: API_KEY environment variable is NOT SET.")
        raise HTTPException(status_code=500, detail="Server configuration error.")
    
    # Use secrets.compare_digest to mitigate timing attacks
    if not secrets.compare_digest(api_key, expected_key):
        raise HTTPException(status_code=403, detail="Invalid or unauthorized API key.")
    return api_key

# --- 3. Security: Strict Input Schema ---
class TransactionFeatures(BaseModel):
    # Defining exact types and constraints for the model
    Avg_min_between_sent_tnx: float = Field(..., ge=0)
    Avg_min_between_received_tnx: float = Field(..., ge=0)
    Time_Diff_between_first_and_last_Mins: float
    Sent_tnx: float
    Received_Tnx: float
    Number_of_Created_Contracts: float
    # Strictly forbid any extra fields not defined in the training set
    model_config = {"extra": "forbid"} 

# --- 4. Load Model securely ---
# FAIL-FAST: Use os.environ here to crash the app immediately if the ID is missing
try:
    RUN_ID = os.environ["MLFLOW_RUN_ID"]
except KeyError:
    logger.critical("DEPLOYMENT ERROR: MLFLOW_RUN_ID is missing from environment.")
    raise RuntimeError("MLFLOW_RUN_ID environment variable is required.")

model_uri = f"runs:/{RUN_ID}/xgb_model"

try:
    model = mlflow.xgboost.load_model(model_uri)
except Exception as e:
    logger.error("Failed to initialize ML model artifact", exc_info=True)
    model = None

# FIXED: Now strictly uses TransactionFeatures instead of 'dict' to trigger FastAPI validation
@app.post("/predict")
def predict_fraud(transaction_data: TransactionFeatures, _: str = Depends(verify_api_key)):
    if model is None:
        raise HTTPException(status_code=500, detail="Prediction engine unavailable.")
    
    try:
        # Pydantic has already validated types and field counts by this point
        df = pd.DataFrame([transaction_data.model_dump()])
        prediction = model.predict(df)
        return {"is_fraud": int(prediction[0])}
    
    except Exception as e:
        logger.error("Runtime prediction error", exc_info=True)
        # Masking internal details from the user
        raise HTTPException(status_code=422, detail="Unable to process transaction data.")

@app.get("/")
def health_check():
    return {"status": "Secure API is online"}