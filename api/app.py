from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import requests
import os
import logging
from pydantic import BaseModel

# Setup logging
logging.basicConfig(filename='api.log', level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load model - ABSOLUTE PATH VERSION
MODEL_PATH = "F:/Projects/deepseek-fraud-detection/ml_model/xgb_model.pkl"
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    raise RuntimeError(f"Could not load ML model: {str(e)}")
# Ollama config
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Input validation
class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

@app.post("/predict")
async def predict(transaction: Transaction):
    try:
        # Prepare input
        input_data = pd.DataFrame([transaction.dict()])[model.feature_names_in_]
        
        # Predict
        proba = model.predict_proba(input_data)[0][1]
        verdict = "High" if proba > 0.7 else "Medium" if proba > 0.3 else "Low"
        
        # Get explanation
        explanation = requests.post(
            OLLAMA_API_URL,
            json={
                "model": "deepseek-r1:1.5b",
                "prompt": f"Explain why transaction Amount=${transaction.Amount} at Time={transaction.Time} is {verdict} risk in 1 sentence.",
                "stream": False
            }
        ).json()["response"]
        
        return {
            "fraud_probability": float(proba),
            "verdict": verdict,
            "explanation": explanation.strip()
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))