import json
import joblib
import pandas as pd
import EDA_Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
from os import getcwd

# 1. Load best hyperparameters from file
with open("best_params.json", "r") as f:
    best_params = json.load(f)

# 2. Load and preprocess test set
df_test = EDA_Pipeline.process_fraud_data(getcwd() + "Data/fraudTest.csv")
X_test = df_test.drop(columns=["is_fraud"])
y_test = df_test["is_fraud"]

# 3. Initialize XGBoost model with best hyperparameters

model = joblib.load("xgb_fraud_model.pkl")

# 5. Predict probabilities and apply threshold
y_proba = model.predict_proba(X_test)[:, 1]
threshold = 0.85
y_pred = (y_proba > threshold).astype(int)

# 6. Evaluate performance
print(f"\n📊 Performance on fraudTest.csv (Threshold = {threshold}):")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:   ", recall_score(y_test, y_pred))
print("F1 Score: ", f1_score(y_test, y_pred))
print("ROC AUC:  ", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Confusion matrix and counts
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("🔍 Confusion Matrix Breakdown:")
print(f"True Positives (TP):  {tp}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Negatives (TN):  {tn}")

# 8. Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Threshold = {threshold})")
plt.tight_layout()
plt.show()

# 9. Compute custom utility
utility = (tp * 50) - (fn * 100) - (fp * 5)
print(f"\n💰 Total Utility: {utility:,}")
