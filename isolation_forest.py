import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

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
df["anomaly_score"] = model.decision_function(X=X_scaled)
df["fraud_score"] = -df["anomaly_score"]

suspicious_transactions = df[df["anomaly_label"] == -1]
print(suspicious_transactions.sort_values("anomaly_score").head(20))
df.to_csv("data/transactions_with_anomalies.csv", index=False)
print("Saved file with anomaly score")

y_true = df["is_fraud"]
y_pred = (df["anomaly_label"] == -1).astype(int)

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))