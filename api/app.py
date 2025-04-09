from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import requests
import os
import logging
import numpy as np
from pydantic import BaseModel
from typing import Dict, Optional
from datetime import datetime

# Setup logging
logging.basicConfig(
    filename='api.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fraud Detection API",
    description="API for detecting fraudulent transactions using XGBoost",
    version="1.1.0"
)

# Configuration
class Config:
    MODEL_PATH = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "ml_model", "xgb_model.pkl"
    ))
    METADATA_PATH = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "ml_model", "metadata.pkl"
    ))
    OLLAMA_URL = "http://localhost:11434/api/generate"
    TIMEOUT = 15

# Load model and metadata
try:
    if not all(os.path.exists(path) for path in [Config.MODEL_PATH, Config.METADATA_PATH]):
        raise FileNotFoundError("Model or metadata file not found")
    
    model = joblib.load(Config.MODEL_PATH)
    metadata = joblib.load(Config.METADATA_PATH)
    logger.info("Model and metadata loaded successfully")
    
except Exception as e:
    logger.error(f"Loading failed: {str(e)}", exc_info=True)
    raise RuntimeError(f"Could not load required files: {str(e)}")

# Data models
class TransactionInput(BaseModel):
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
    additional_info: Optional[Dict] = None

class PredictionOutput(BaseModel):
    fraud_probability: float
    verdict: str
    explanation: str
    timestamp: str
    model_version: str = "1.1"

# Helper functions
def standardize_amount_time(amount: float, time: float) -> Dict:
    """Standardize amount and time using training statistics"""
    return {
        'Amount_Std': (amount - metadata['amount_mean']) / metadata['amount_std'],
        'Time_Std': (time - metadata['time_mean']) / metadata['time_std']
    }

def prepare_features(transaction: Dict) -> pd.DataFrame:
    """Prepare input features for prediction"""
    try:
        # Standardize amount and time
        std_features = standardize_amount_time(transaction['Amount'], transaction['Time'])
        
        # Combine all features
        features = {
            **{f'V{i}': transaction[f'V{i}'] for i in range(1, 29)},
            **std_features
        }
        
        # Ensure correct feature order
        return pd.DataFrame([features])[metadata['features']]
        
    except Exception as e:
        logger.error(f"Feature preparation failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Feature processing error")

def get_llm_explanation(transaction: Dict, proba: float, verdict: str) -> str:
    """Get explanation from DeepSeek"""
    try:
        prompt = f"""
        Transaction Risk Analysis:
        
        Context:
        - Amount: ${transaction['Amount']:.2f}
        - Time: {transaction['Time']} seconds
        - Probability: {proba:.6f}
        - Key Features:
          V1: {transaction['V1']:.2f} (PCA1)
          V2: {transaction['V2']:.2f} (PCA2) 
          V3: {transaction['V3']:.2f} (PCA3)
        
        Task:
        Explain the {verdict} risk prediction in 2-3 concise sentences:
        1. Highlight the most suspicious values
        2. Explain their significance
        3. Relate to known fraud patterns
        
        Technical Requirements:
        - Use financial fraud terminology
        - Reference specific feature values
        - Avoid vague statements
        - Maximum 3 sentences
        """
        
        response = requests.post(
            Config.OLLAMA_URL,
            json={
                "model": "deepseek-r1:1.5b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "max_tokens": 150
                }
            },
            timeout=Config.TIMEOUT
        )
        response.raise_for_status()
        return response.json().get("response", "Explanation unavailable").strip()
        
    except Exception as e:
        logger.warning(f"LLM explanation failed: {str(e)}")
        return "Could not generate explanation"

def determine_verdict(proba: float) -> str:
    """Determine risk level based on configured thresholds"""
    if proba >= metadata['thresholds']['high']:
        return "High"
    elif proba >= metadata['thresholds']['medium']:
        return "Medium"
    return "Low"

# API endpoints
@app.post("/predict", response_model=PredictionOutput)
async def predict_fraud(transaction: TransactionInput):
    """Main prediction endpoint"""
    try:
        # Prepare input features
        df_input = prepare_features(transaction.dict())
        
        # Predict
        proba = float(model.predict_proba(df_input)[0,1])
        verdict = determine_verdict(proba)
        
        # Get explanation
        explanation = get_llm_explanation(
            transaction=transaction.dict(),
            proba=proba,
            verdict=verdict
        )
        
        # Log prediction
        logger.info(
            f"Prediction - Amount: ${transaction.Amount:.2f}, "
            f"Prob: {proba:.6f}, Verdict: {verdict}"
        )
        
        return {
            "fraud_probability": proba,
            "verdict": verdict,
            "explanation": explanation,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction"
        )

@app.post("/debug-predict")
async def debug_prediction(transaction: TransactionInput):
    """Debug endpoint to inspect feature processing"""
    try:
        df_input = prepare_features(transaction.dict())
        
        return {
            "processed_features": df_input.iloc[0].to_dict(),
            "feature_importances": dict(zip(
                model.feature_names_in_,
                model.feature_importances_
            )),
            "model_thresholds": metadata['thresholds']
        }
        
    except Exception as e:
        logger.error(f"Debug failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def get_model_info():
    """Get model metadata"""
    return {
        "features": metadata['features'],
        "thresholds": metadata['thresholds'],
        "scaling_params": {
            "amount": {"mean": metadata['amount_mean'], "std": metadata['amount_std']},
            "time": {"mean": metadata['time_mean'], "std": metadata['time_std']}
        },
        "model_version": "1.1"
    }