import os
import secrets
import logging
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import mlflow.pyfunc  # Change this from mlflow.xgboost
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ethereum Fraud Detection API")

# --- 1. Security: CORS Middleware ---
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != [""] else ["*"],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# --- 2. Security: API Key Authentication ---
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

def verify_api_key(api_key: str = Security(api_key_header)):
    expected_key = os.environ.get("API_KEY") 
    if not expected_key:
        logger.critical("SECURITY CONFIG ERROR: API_KEY is NOT SET.")
        raise HTTPException(status_code=500, detail="Server configuration error.")
    
    if not secrets.compare_digest(api_key, expected_key):
        raise HTTPException(status_code=403, detail="Invalid or unauthorized API key.")
    return api_key

# --- 3. Security: Strict Input Schema ---
class TransactionFeatures(BaseModel):
    Avg_min_between_sent_tnx: float = Field(..., ge=0)
    Avg_min_between_received_tnx: float = Field(..., ge=0)
    Time_Diff_between_first_and_last_Mins: float
    Sent_tnx: float
    Received_Tnx: float
    Number_of_Created_Contracts: float
    model_config = {"extra": "forbid"} 

# --- 4. Load Model securely ---
mlflow.set_tracking_uri("file:./mlruns")
RUN_ID = os.environ.get("MLFLOW_RUN_ID")

if not RUN_ID:
    logger.critical("DEPLOYMENT ERROR: MLFLOW_RUN_ID is missing.")
    model = None
else:
    # We point directly to the artifacts folder based on your screenshot structure
    # Path: mlruns/1/models/m-<RUN_ID>/artifacts
    model_path = os.path.join(os.getcwd(), "mlruns", "1", "models", f"m-{RUN_ID}", "artifacts")
    
    try:
        logger.info(f"Attempting to load model from: {model_path}")
        # Use pyfunc instead of xgboost for better compatibility
        model = mlflow.pyfunc.load_model(model_path)
        logger.info("✅ Model loaded successfully via pyfunc!")
    except Exception as e:
        logger.error(f"❌ Failed to load model at {model_path}: {e}")
        model = None

@app.post("/predict")
def predict_fraud(transaction_data: TransactionFeatures, _: str = Depends(verify_api_key)):
    if model is None:
        raise HTTPException(status_code=500, detail="Prediction engine unavailable. Check logs.")
    
    try:
        df = pd.DataFrame([transaction_data.model_dump()])
        prediction = model.predict(df)
        return {"is_fraud": int(prediction[0])}
    except Exception as e:
        logger.error("Runtime prediction error", exc_info=True)
        raise HTTPException(status_code=422, detail="Unable to process transaction data.")

@app.get("/")
def health_check():
    return {"status": "Secure API is online", "model_loaded": model is not None}