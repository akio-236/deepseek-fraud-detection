import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
import joblib

# Load data
df = pd.read_csv("../data/creditcard.csv")
X, y = df.drop('Class', axis=1), df['Class']

# Scale features (RobustScaler handles outliers)
scaler = RobustScaler()
X[['Amount', 'Time']] = scaler.fit_transform(X[['Amount', 'Time']])

# Balance classes using SMOTE (synthetic minority oversampling)
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Split data (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Train XGBoost (optimized for imbalanced data)
model = XGBClassifier(scale_pos_weight=100, eval_metric='aucpr')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_pred))

# Save the model
joblib.dump(model, "xgb_model.pkl")
print("Model saved as 'xgb_model.pkl'")