import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import joblib
import mlflow
import mlflow.sklearn

data_path = Path("data/transaction_features.csv")
output_path = Path("data/transactions_with_anomalies.csv")
model_path = Path("models/isolation_forest.pkl")
score_plot_path = Path("data/anomaly_score_distribution.png")
roc_plot_path = Path("data/roc_curve.png")

model_path.parent.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("fraud_anomaly_detection")

df = pd.read_csv(data_path, encoding="utf-8")

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
y_true = df["is_fraud"]

with mlflow.start_run(run_name="isolation_forest_baseline"):

    params = {
        "model_type": "IsolationForest",
        "n_estimators": 200,
        "contamination": 0.15,
        "random_state": 42,
        "scaler": "StandardScaler",
        "num_features": len(features),
        "features": ",".join(features),
    }

    mlflow.log_params(params)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", IsolationForest(
            n_estimators=200,
            contamination=0.15,
            random_state=42
        ))
    ])

    pipeline.fit(X)

    model = pipeline.named_steps["model"]
    df["anomaly_label"] = pipeline.predict(X)
    df["anomaly_score"] = pipeline.decision_function(X)
    df["fraud_score"] = -df["anomaly_score"]
    y_score = df["fraud_score"]
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    print("AUC:", auc)

    if auc >= 0.82:
        print("Target met")
    else:
        print("Target not met")

    mlflow.log_metric("auc", auc)
    mlflow.log_metric("score_min", df["fraud_score"].min())
    mlflow.log_metric("score_max", df["fraud_score"].max())
    mlflow.log_metric("num_transactions", len(df))
    mlflow.log_metric("num_suspicious_transactions", int((df["anomaly_label"] == -1).sum()))

    joblib.dump(
        {
            "pipeline": pipeline,
            "features": features,
            "score_min": df["fraud_score"].min(),
            "score_max": df["fraud_score"].max(),
            "auc": auc,
        },
        model_path
    )

    df.to_csv(output_path, index=False)
    print("Saved file with anomaly score")

    suspicious_transactions = df[df["anomaly_label"] == -1]
    print(suspicious_transactions.sort_values("anomaly_score").head(20))

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
    plt.savefig(score_plot_path, dpi=300)
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
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()

    mlflow.log_artifact(str(model_path))
    mlflow.log_artifact(str(output_path))
    mlflow.log_artifact(str(score_plot_path))
    mlflow.log_artifact(str(roc_plot_path))

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        registered_model_name="IsolationForestFraudDetector"
    )

    print("Saved plots:")
    print(score_plot_path)
    print(roc_plot_path)
    print("MLflow run logged successfully")