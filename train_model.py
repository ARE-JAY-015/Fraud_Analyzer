import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print("Loading dataset...")
df = pd.read_csv("dataset/fraud_transactions.csv")

# 1. Feature Engineering
df["amount_to_avg_ratio"] = df["amount"] / (df["avg_amount"] + 1e-5)

# Drop ID and target
X_df = df.drop(columns=["transaction_id", "fraud"])
y_series = df["fraud"]

# One-hot encode categorical features
X_df = pd.get_dummies(X_df, columns=["merchant_category", "payment_method"], drop_first=True)
feature_names = X_df.columns.tolist()

X = np.asarray(X_df.values, dtype=np.float64)
y = np.asarray(y_series.values, dtype=np.int64)

# 2. Pure NumPy Stratified Train-Test Split (Bypasses Python 3.14 metadata bug)
print("Splitting dataset (80% train, 20% test)...")
np.random.seed(42)

idx_0 = np.where(y == 0)[0]
idx_1 = np.where(y == 1)[0]

np.random.shuffle(idx_0)
np.random.shuffle(idx_1)

split_0 = int(len(idx_0) * 0.8)
split_1 = int(len(idx_1) * 0.8)

train_idx = np.concatenate([idx_0[:split_0], idx_1[:split_1]])
test_idx = np.concatenate([idx_0[split_0:], idx_1[split_1:]])

np.random.shuffle(train_idx)
np.random.shuffle(test_idx)

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# 3. Train balanced Random Forest
print("Training Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    class_weight="balanced", 
    random_state=42,
    n_jobs=1
)
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 55)
print("FRAUDGUARD: MODEL EVALUATION METRICS")
print("=" * 55)
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"  Legitimate Approved (True Negatives)  : {cm[0][0]}")
print(f"  Legitimate Blocked  (False Positives) : {cm[0][1]}")
print(f"  Fraud Missed        (False Negatives) : {cm[1][0]}")
print(f"  Fraud Prevented     (True Positives)  : {cm[1][1]}")

print("\nDetailed Performance Report:")
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

# 5. Top Risk Indicators
importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
print("\nTop 5 Fraud Risk Indicators:")
for feat, imp in importances.head(5).items():
    print(f"  - {feat:<25}: {imp * 100:.2f}% contribution")

# 6. Save model artifacts
with open("fraud_model.pkl", "wb") as f:
    pickle.dump({"model": model, "columns": feature_names}, f)

print("\nArtifact successfully saved to fraud_model.pkl")
print("=" * 55)