import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

# Load data
df = pd.read_csv("../data/creditcard.csv")

# Feature engineering
df['Amount_Std'] = RobustScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
df['Time_Std'] = RobustScaler().fit_transform(df['Time'].values.reshape(-1, 1))

# Prepare features
features = [f'V{i}' for i in range(1,29)] + ['Amount_Std', 'Time_Std']
X = df[features]
y = df['Class']

# Handle class imbalance
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Train model with better parameters
model = XGBClassifier(
    scale_pos_weight=100,
    eval_metric='aucpr',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred):.4f}")

# Save model
os.makedirs("../ml_model", exist_ok=True)
joblib.dump(model, "../ml_model/xgb_model.pkl")
print("Model saved to ../ml_model/xgb_model.pkl")