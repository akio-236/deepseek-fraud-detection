import pandas as pd
import numpy as np
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (roc_auc_score, classification_report, 
                            average_precision_score, confusion_matrix,
                            precision_recall_curve)
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
import joblib
import os
import matplotlib.pyplot as plt
import logging

def configure_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )

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
    
    # Conservative model parameters to prevent overfitting
    model = XGBClassifier(
    scale_pos_weight=200,  # More aggressive weighting
    eval_metric='aucpr',
    max_depth=8,
    learning_rate=0.05,
    n_estimators=1000,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=0.5,
    min_child_weight=5,  # Important for imbalanced data
    gamma=0.1,
    early_stopping_rounds=50,
    random_state=42
)
    
    # Manual cross-validation without early stopping
    cv = StratifiedKFold(n_splits=5)
    cv_scores = []
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Balance only the training fold
        sm = SMOTE(random_state=42)
        X_bal, y_bal = sm.fit_resample(X_train_cv, y_train_cv)
        
        model.fit(
            X_bal, y_bal,
            sample_weight=compute_sample_weight('balanced', y_bal)
        )
        
        y_proba = model.predict_proba(X_val_cv)[:,1]
        cv_scores.append(average_precision_score(y_val_cv, y_proba))
    
    logging.info(f"Cross-val AP scores: {cv_scores}")
    logging.info(f"Mean AP: {np.mean(cv_scores):.4f}")
    
    # Final training on full balanced dataset
    sm = SMOTE(random_state=42)
    X_bal_full, y_bal_full = sm.fit_resample(X_train, y_train)
    
    final_model = XGBClassifier(
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
        eval_metric='aucpr',
        max_depth=4,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=2.0,
        reg_lambda=2.0,
        min_child_weight=5,
        gamma=0.2,
        random_state=42
    )
    
    final_model.fit(
        X_bal_full, y_bal_full,
        sample_weight=compute_sample_weight('balanced', y_bal_full),
        verbose=10
    )
    
    return final_model

def evaluate_model(model, X_test, y_test):
    """Comprehensive model evaluation"""
    logging.info("\nEvaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    
    # Classification report
    logging.info("\nClassification Report:")
    logging.info(classification_report(y_test, y_pred))
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('../ml_model/pr_curve.png')
    
    # Metrics
    logging.info(f"\nAUPRC: {average_precision_score(y_test, y_proba):.4f}")
    logging.info(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logging.info("\nConfusion Matrix:")
    logging.info(cm)
    
    # Feature importance
    plt.figure(figsize=(12,8))
    plot_importance(model, max_num_features=20)
    plt.tight_layout()
    plt.savefig('../ml_model/feature_importance.png')
    logging.info("\nSaved evaluation plots")

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
        },
        'model_config': {
            'scale_pos_weight': len(df[df['Class']==0])/len(df[df['Class']==1]),
            'max_depth': 4,
            'learning_rate': 0.05
        }
    }
    
    os.makedirs("../ml_model", exist_ok=True)
    joblib.dump(model, "../ml_model/xgb_model.pkl")
    joblib.dump(metadata, "../ml_model/metadata.pkl")
    logging.info("\nModel and metadata saved successfully")

def main():
    configure_logging()
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
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate on original imbalanced test set
        evaluate_model(model, X_test, y_test)
        
        # Save artifacts
        save_model_and_metadata(model, df)
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()