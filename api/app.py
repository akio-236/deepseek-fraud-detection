from fastapi import FastAPI
import joblib
import pandas as pd
import requests

app = FastAPI()

# Load the trained model
model = joblib.load("../model/xgb_model.pkl")

# Ollama API setup (for explanations)
OLLAMA_API_URL = "http://localhost:11434/api/generate"

def explain_verdict(transaction: dict, verdict: str) -> str:
    """Ask DeepSeek-R1 to explain the fraud risk."""
    prompt = f"""
    Transaction Details: 
    - Amount: ${transaction['Amount']} 
    - Time: {transaction['Time']} seconds
    - Features: V1-V28 (anonymized PCA components)
    
    The ML model predicted this as '{verdict}' risk. Explain why in 1 sentence.
    """
    response = requests.post(
        OLLAMA_API_URL,
        json={"model": "deepseek-r1:1.5b", "prompt": prompt, "stream": False}
    )
    return response.json()["response"]

@app.post("/predict")
def predict(transaction: dict):
    # Convert input to DataFrame
    df = pd.DataFrame([transaction])
    
    # Predict fraud probability
    proba = model.predict_proba(df)[0][1]  # P(fraud)
    
    # Classify risk
    verdict = (
        "High" if proba > 0.7 
        else "Medium" if proba > 0.3 
        else "Low"
    )
    
    # Get explanation from DeepSeek
    explanation = explain_verdict(transaction, verdict)
    
    return {
        "fraud_probability": float(proba),
        "verdict": verdict,
        "explanation": explanation.strip()
    }