import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from EDA_Pipeline import process_fraud_data, oversample
from os import getcwd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
# from PCA import perform_PCA


def load_data(path):
    # Load dataset
    df = process_fraud_data(getcwd() + path)
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']

    # print(f"Total Data: {len(y)}")
    # print(f"Fraud Cases: {np.count_nonzero(y)}")
    # print(f"Percent Fraud: {np.count_nonzero(y) / len(y) * 100:.4f}%\n")
    
    # Standardize features using z-scores
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Add intercept for logistic regression
    X_scaled = sm.add_constant(X_scaled)
    return X_scaled, y


# def normalize_numerical_columns(X):
#     df_normalized = X.copy()
#     for col in X.select_dtypes(include=['number']).columns:
#         min_val = X[col].min()
#         max_val = X[col].max()
#         if min_val != max_val:  # Avoid division by zero
#             df_normalized[col] = (X[col] - min_val) / (max_val - min_val)
#         else:
#             df_normalized[col] = 0  # Set to 0 if all values are the same
#     return df_normalized


def backward_elimination(X, y, threshold=0.05):
    X = X.copy()  # Avoid modifying the original dataframe
    while True:
        model = sm.Logit(y, X).fit(disp=0)  # Fit logistic regression
        p_values = model.pvalues[1:]  # Exclude intercept
        max_p_value = p_values.max()  # Find max p-value
        
        if max_p_value < threshold:
            break  # Stop if all p-values are below threshold
        
        worst_feature = p_values.idxmax()  # Get feature with highest p-value
        X.drop(columns=[worst_feature], inplace=True)  # Remove worst feature
        print(f"Dropping feature: {worst_feature} with p_value: {max_p_value}")
    
    return X, model


def cross_validation(X, y, n_splits=5, threshold=0.85):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    best_model = None
    best_accuracy = 0
    best_features = None

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Perform backward elimination on training set
        X_selected, model = backward_elimination(X_train, y_train)
        # model = sm.Logit(y_train, X_train).fit(disp=0)

        # Ensure validation data has the same selected features
        X_val = X_val[X_selected.columns]
        # X_val = X_val[X_train.columns]

        # Make predictions
        y_pred = (model.predict(X_val) >= threshold).astype(int)

        # Compute accuracy
        accuracy = accuracy_score(y_val, y_pred)
        
        # Store the best model based on accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_features = X_selected.columns  # Save selected features
            # best_features = X_train.columns

    print(f"\n✅ Best Model Accuracy: {best_accuracy:.4f}\n")
    return best_model, best_features


def main():
    # Load and preprocess data
    X_train, y_train = load_data("/Data/fraudTrain.csv")
    X_train, y_train = oversample(X_train, y_train)
    # pca, X_train, _ = perform_PCA(X_train, n_components=15)
    
    # Perform backward elimination
    # X_selected, final_model = backward_elimination(X_train, y_train)
    final_model, selected_features = cross_validation(X_train, y_train)
    # final_model, selected_features = cross_validation(X_selected, y_train)
    
    # Print selected features
    print("Selected Features:")
    for f in selected_features:
        print(f)
    print()

    threshold = 0.95

    X_test, y_test = load_data("/Data/fraudTest.csv")
    # X_test = X_test[selected_features]
    # X_test = pd.DataFrame(pca.transform(X_test))
    X_test = X_test[selected_features]
    y_pred = (final_model.predict(X_test) >= threshold).astype(int)

    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Test Accuracy: {test_accuracy * 100:.2f}%\n")

    # Evaluate performance
    print(f"\n📊 Performance on fraudTest.csv")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:   ", recall_score(y_test, y_pred))
    print("F1 Score: ", f1_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion matrix and counts
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("🔍 Confusion Matrix Breakdown:")
    print(f"True Positives (TP):  {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Negatives (TN):  {tn}")

    # Compute custom utility
    utility = (tp * 50) - (fn * 100) - (fp * 5)
    print(f"\n💰 Total Utility: {utility:,}")

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix")
    plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":
    main()
