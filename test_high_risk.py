import requests
import json

# Should definitely be flagged as high risk
TEST_TRANSACTION = {
    "Time": 0,
    "Amount": 5000,
    "V1": -5.0, "V2": 3.5, "V3": -4.2,
    "V4": 1.5, "V5": -2.7, "V6": 3.1,
    "V7": -1.3, "V8": 1.8, "V9": -3.5,
    "V10": 1.9, "V11": -1.6, "V12": 1.4,
    "V13": -1.2, "V14": 1.1, "V15": -2.8,
    "V16": 1.3, "V17": -2.4, "V18": 1.7,
    "V19": -3.9, "V20": 1.2, "V21": -1.1,
    "V22": 1.6, "V23": -2.5, "V24": 1.0,
    "V25": -1.3, "V26": 1.5, "V27": -2.7,
    "V28": 1.4
}

def test_transaction():
    print("Testing high-risk transaction...")
    response = requests.post(
        "http://localhost:8000/inspect-features",
        json=TEST_TRANSACTION
    )
    result = response.json()
    
    print("\nFeature Importances:")
    importances = sorted(
        result['feature_importances'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    for feat, imp in importances[:10]:
        print(f"{feat}: {imp:.4f}")
    
    print(f"\nPredicted Fraud Probability: {result['prediction_probability']:.6f}")
    
    if result['prediction_probability'] > 0.65:
        print("✅ Correctly identified as high risk")
    else:
        print("❌ Failed - should be high risk")

if __name__ == "__main__":
    test_transaction()