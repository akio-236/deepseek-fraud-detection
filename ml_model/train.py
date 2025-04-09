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

def load_and_prepare_data():
    """Load and prepare the dataset"""
    logging.info("Loading data...")
    df = pd.read_csv("../data/creditcard.csv")
    logging.info(f"Data shape: {df.shape}")
    
    # Analyze class distribution
    class_dist = df['Class'].value_counts(normalize=True)
    logging.info(f"\nClass distribution:\n{class_dist}")
    
    # Feature engineering
    logging.info("\nEngineering features...")
    scaler = RobustScaler()
    df['Amount_Std'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['Time_Std'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    
    return df

def train_model(X_train, y_train):
    """Train XGBoost model with proper validation"""
    logging.info("\nTraining model...")
    
    # Remove early stopping for cross-validation
    model = XGBClassifier(
        scale_pos_weight=100,
        eval_metric='aucpr',
        max_depth=6,
        learning_rate=0.1,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42
    )
    
    # Cross-validation without early stopping
    cv = StratifiedKFold(n_splits=5)
    cv_scores = cross_val_score(
        model, X_train, y_train, 
        cv=cv, scoring='average_precision'
    )
    logging.info(f"Cross-val AP scores: {cv_scores}")
    logging.info(f"Mean AP: {np.mean(cv_scores):.4f}")
    
    # Final training with early stopping
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    final_model = XGBClassifier(
        scale_pos_weight=100,
        eval_metric='aucpr',
        max_depth=6,
        learning_rate=0.1,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        early_stopping_rounds=50,
        random_state=42
    )
    
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=10
    )
    
    return final_model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    logging.info("\nEvaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    
    logging.info("\nClassification Report:")
    logging.info(classification_report(y_test, y_pred))
    
    logging.info(f"\nAUPRC: {average_precision_score(y_test, y_proba):.4f}")
    logging.info(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(12,8))
    plot_importance(model, max_num_features=20)
    plt.tight_layout()
    os.makedirs("../ml_model", exist_ok=True)
    plt.savefig('../ml_model/feature_importance.png')
    logging.info("\nSaved feature importance plot")

def save_model_and_metadata(model, df):
    """Save model and required metadata"""
    metadata = {
        'features': list(model.feature_names_in_),
        'amount_mean': float(df['Amount'].mean()),
        'amount_std': float(df['Amount'].std()),
        'time_mean': float(df['Time'].mean()),
        'time_std': float(df['Time'].std()),
        'thresholds': {
            'high': 0.85,
            'medium': 0.5,
            'low': 0.0
        }
    }
    
    joblib.dump(model, "../ml_model/xgb_model.pkl")
    joblib.dump(metadata, "../ml_model/metadata.pkl")
    logging.info("\nModel and metadata saved successfully")

def main():
    try:
        # Load and prepare data
        df = load_and_prepare_data()
        
        # Prepare features
        features = [f'V{i}' for i in range(1,29)] + ['Amount_Std', 'Time_Std']
        X = df[features]
        y = df['Class']
        
        # Split data (maintain original distribution in test set)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Balance only the training data
        sm = SMOTE(random_state=42)
        X_bal, y_bal = sm.fit_resample(X_train, y_train)
        logging.info(f"\nBalanced training data shape: {X_bal.shape}")
        
        # Train model
        model = train_model(X_bal, y_bal)
        
        # Evaluate on original imbalanced test set
        evaluate_model(model, X_test, y_test)
        
        # Save artifacts
        save_model_and_metadata(model, df)
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )
    main()