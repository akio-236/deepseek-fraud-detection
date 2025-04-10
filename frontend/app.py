import pandas as pd
import numpy as np
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, classification_report, 
                            average_precision_score, confusion_matrix)
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
import joblib
import os
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_prepare_data():
    """Load and prepare the dataset"""
    logger.info("Loading data...")
    df = pd.read_csv("../data/creditcard.csv")
    logger.info(f"Data shape: {df.shape}")
    
    # Analyze class distribution
    class_dist = df['Class'].value_counts(normalize=True)
    logger.info(f"\nClass distribution:\n{class_dist}")
    
    # Feature engineering
    logger.info("\nEngineering features...")
    scaler = RobustScaler()
    df['Amount_Std'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['Time_Std'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    
    return df

def train_model(X_train, y_train, X_val=None, y_val=None):
    """Train XGBoost model with proper validation"""
    logger.info("\nTraining model...")
    
    model = XGBClassifier(
        scale_pos_weight=200,
        eval_metric='aucpr',
        max_depth=8,
        learning_rate=0.05,
        n_estimators=1000,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=0.5,
        min_child_weight=5,
        gamma=0.1,
        early_stopping_rounds=50 if X_val is not None else None,
        random_state=42
    )
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5)
    cv_scores = cross_val_score(
        model, X_train, y_train, 
        cv=cv, scoring='average_precision'
    )
    logger.info(f"Cross-val AP scores: {cv_scores}")
    logger.info(f"Mean AP: {np.mean(cv_scores):.4f}")
    
    # Fit with or without validation
    if X_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=10
        )
    else:
        model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    logger.info("\nEvaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    logger.info(f"\nAUPRC: {average_precision_score(y_test, y_proba):.4f}")
    logger.info(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(12,8))
    plot_importance(model, max_num_features=20)
    plt.tight_layout()
    os.makedirs("../ml_model", exist_ok=True)
    plt.savefig('../ml_model/feature_importance.png')
    logger.info("\nSaved feature importance plot")

def save_model_and_metadata(model, df):
    """Save model and required metadata"""
    metadata = {
        'features': list(model.feature_names_in_),
        'amount_mean': float(df['Amount'].mean()),
        'amount_std': float(df['Amount'].std()),
        'time_mean': float(df['Time'].mean()),
        'time_std': float(df['Time'].std()),
        'thresholds': {
            'high': 0.65,
            'medium': 0.35,
            'low': 0.0
        }
    }
    
    joblib.dump(model, "../ml_model/xgb_model.pkl")
    joblib.dump(metadata, "../ml_model/metadata.pkl")
    logger.info("\nModel and metadata saved successfully")

def main():
    try:
        # Load and prepare data
        df = load_and_prepare_data()
        
        # Prepare features
        features = [f'V{i}' for i in range(1,29)] + ['Amount_Std', 'Time_Std']
        X = df[features]
        y = df['Class']
        
        # Split data (maintain original distribution in validation)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Balance only the training data
        sm = SMOTE(random_state=42)
        X_bal, y_bal = sm.fit_resample(X_train, y_train)
        logger.info(f"\nBalanced training data shape: {X_bal.shape}")
        
        # Train model with validation
        model = train_model(X_bal, y_bal, X_val, y_val)
        
        # Evaluate on original imbalanced test set
        evaluate_model(model, X_test, y_test)
        
        # Save artifacts
        save_model_and_metadata(model, df)
        
        # Test on known fraud cases
        fraud_cases = df[df['Class']==1].sort_values('Amount', ascending=False).head(5)
        logger.info("\nTesting on known fraud cases:")
        for _, case in fraud_cases.iterrows():
            proba = model.predict_proba([case[features]])[0,1]
            logger.info(f"Amount: ${case['Amount']}, Prob: {proba:.6f}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()