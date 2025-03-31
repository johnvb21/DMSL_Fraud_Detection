from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from scipy.stats import uniform, randint
import xgboost as xgb
import EDA_pipeline
import json
import joblib


# 1. Load and preprocess data
df = EDA_pipeline.process_fraud_data("fraudTrain.csv")
X = df.drop(columns=['is_fraud'])
y = df['is_fraud']

# 2. Define base model (will be tuned)
base_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42
)

# 3. Define hyperparameter search space
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.7, 0.3),
    'scale_pos_weight': [ (y == 0).sum() / (y == 1).sum() ]  # imbalance handling
}

# 4. Set up RandomizedSearchCV with 5-fold cross-validation
random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=10,               # number of random combinations
    scoring='roc_auc',       # optimize AUC
    cv=5,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# 5. Fit the model
random_search.fit(X, y)

# 6. Print best results
print("\n✅ Best ROC AUC Score from CV:", random_search.best_score_)
print("🏆 Best Hyperparameters:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")
# 7. Save best hyperparameters to JSON

with open("best_params.json", "w") as f:
    json.dump(random_search.best_params_, f, indent=4)

# 8. Save trained model to file
joblib.dump(random_search.best_estimator_, "xgb_fraud_model.pkl")

print("\n📁 Saved best parameters to 'best_params.json'")
print("📦 Saved trained model to 'xgb_fraud_model.pkl'")