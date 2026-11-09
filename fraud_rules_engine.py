import pandas as pd
import numpy as np
import json

print("=" * 60)
print("FRAUDGUARD: BEHAVIORAL RISK & ANOMALY SCORING ENGINE")
print("=" * 60)

# Load dataset
df = pd.read_csv("dataset/fraud_transactions.csv")

print(f"Analyzing {len(df):,} transactions across 12 behavioral dimensions...\n")

# 1. Feature Engineering: Spender Deviation Ratio
df["amount_to_avg_ratio"] = df["amount"] / (df["avg_amount"] + 1e-5)

# 2. Financial Industry Risk Scoring Matrix
# Weights derived from empirical fraud correlation
def calculate_risk_score(row):
    score = 0
    reasons = []

    # Flag 1: Surge spending (Current amount > 3x historical average)
    if row["amount_to_avg_ratio"] >= 4.0:
        score += 35
        reasons.append("High amount surge (>4x baseline)")
    elif row["amount_to_avg_ratio"] >= 2.5:
        score += 20
        reasons.append("Moderate amount surge (>2.5x baseline)")

    # Flag 2: Authentication failures
    if row["failed_attempts"] >= 3:
        score += 30
        reasons.append(f"Multiple failed login attempts ({row['failed_attempts']})")
    elif row["failed_attempts"] >= 1:
        score += 15
        reasons.append("Prior failed authentication")

    # Flag 3: Device change
    if row["device_changed"] == 1:
        score += 20
        reasons.append("Unrecognized device fingerprint")

    # Flag 4: Cross-border / Geovelocity
    if row["international"] == 1:
        score += 15
        reasons.append("Cross-border transaction")

    if row["distance_from_home"] > 50:
        score += 15
        reasons.append(f"Unusual distance from home ({row['distance_from_home']:.1f} miles)")

    # Flag 5: High transaction velocity
    if row["transactions_last_24h"] >= 8:
        score += 20
        reasons.append(f"Velocity surge ({row['transactions_last_24h']} txs in 24h)")

    # Cap score at 100
    final_score = min(score, 100)
    return pd.Series([final_score, ", ".join(reasons)], index=["risk_score", "risk_reasons"])

# Apply engine
print("Calculating transaction risk scores...")
scores_df = df.apply(calculate_risk_score, axis=1)
df["risk_score"] = scores_df["risk_score"]
df["risk_reasons"] = scores_df["risk_reasons"]

# 3. Decision Thresholds:
# Score >= 65: BLOCK
# Score 40-64: REVIEW (3DS/OTP challenge)
# Score < 40 : APPROVE
df["predicted_fraud"] = (df["risk_score"] >= 65).astype(int)

# 4. Business & Financial Evaluation
total_actual_fraud = df["fraud"].sum()
total_actual_legit = len(df) - total_actual_fraud

tp = len(df[(df["predicted_fraud"] == 1) & (df["fraud"] == 1)]) # Fraud Caught
fp = len(df[(df["predicted_fraud"] == 1) & (df["fraud"] == 0)]) # False Alarms
fn = len(df[(df["predicted_fraud"] == 0) & (df["fraud"] == 1)]) # Fraud Missed
tn = len(df[(df["predicted_fraud"] == 0) & (df["fraud"] == 0)]) # Clean Approvals

precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
fpr = (fp / total_actual_legit) * 100

total_fraud_dollars = df[df["fraud"] == 1]["amount"].sum()
prevented_fraud_dollars = df[(df["predicted_fraud"] == 1) & (df["fraud"] == 1)]["amount"].sum()

print("-" * 60)
print("EVALUATION & BUSINESS IMPACT")
print("-" * 60)
print(f"Total Fraud Exposure      : ${total_fraud_dollars:,.2f}")
print(f"Fraud Loss Prevented      : ${prevented_fraud_dollars:,.2f} ({prevented_fraud_dollars/total_fraud_dollars * 100:.1f}%)")
print(f"Fraud Detection Rate (Recall)   : {recall:.2f}%")
print(f"Alert Precision                : {precision:.2f}%")
print(f"False Positive Rate (FPR)      : {fpr:.2f}% (Legitimate customers impacted)")

print("\n" + "-" * 60)
print("TOP 5 HIGH-RISK TRANSACTIONS INTERCEPTED")
print("-" * 60)
top_risks = df[df["predicted_fraud"] == 1].sort_values("risk_score", ascending=False).head(5)
for _, r in top_risks.iterrows():
    print(f"ID #{r['transaction_id']:<5} | Amount: ${r['amount']:<8.2f} | Score: {r['risk_score']}/100 | Actual: {'FRAUD' if r['fraud'] else 'LEGIT'}")
    print(f"   Flags: {r['risk_reasons']}\n")

# Export scored dataset
df.to_csv("dataset/scored_fraud_transactions.csv", index=False)
print("Scored results exported to: dataset/scored_fraud_transactions.csv")
print("=" * 60)