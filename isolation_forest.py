import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("data/transaction_features.csv", encoding="utf-8")

features = [
    "amount_zscore",
    "hour_of_day",
    "day_of_week",
    "txn_count_in_last_1h",
    "txn_count_in_last_24h",
    "amount_vs_user_mean",
    "is_new_merchant"
]

X = df[features]
X = X.fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X=X)

model = IsolationForest(
    n_estimators=200,
    contamination=0.15,
    random_state=42
)

df["anomaly_label"] = model.fit_predict(X=X_scaled)

joblib.dump({
    "model": model,
    "scaler": scaler,
    "features": features
}, "models/isolation_forest.pkl"
)

df["anomaly_score"] = model.decision_function(X=X_scaled)
df["fraud_score"] = -df["anomaly_score"]

suspicious_transactions = df[df["anomaly_label"] == -1]
print(suspicious_transactions.sort_values("anomaly_score").head(20))
df.to_csv("data/transactions_with_anomalies.csv", index=False)
print("Saved file with anomaly score")

y_true = df["is_fraud"]
y_score = df["fraud_score"]

auc = roc_auc_score(y_true=y_true, y_score=y_score)
print("AUC:" , auc)

if auc >= 0.82:
    print("Target met")
else:
    print("Target not met")

plt.figure(figsize=(10, 6))

plt.hist(
    df[df["is_fraud"] == 0]["fraud_score"],
    bins=50,
    alpha=0.6,
    label="Legit",
    density=True
)

plt.hist(
    df[df["is_fraud"] == 1]["fraud_score"],
    bins=50,
    alpha=0.6,
    label="Fraud",
    density=True
)

plt.xlabel("Fraud Score Higher = More Suspicious")
plt.ylabel("Density")
plt.title(f"Fraud Score Distribution: Fraud vs Legit | AUC = {auc:.3f}")
plt.legend()
plt.tight_layout()
plt.savefig("data/anomaly_score_distribution.png", dpi=300)
plt.close()

fpr, tpr, thresholds = roc_curve(y_true, y_score)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Isolation Forest AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("data/roc_curve.png", dpi=300)
plt.close()

print("Saved plots:")
print("data/anomaly_score_distribution.png")
print("data/roc_curve.png")