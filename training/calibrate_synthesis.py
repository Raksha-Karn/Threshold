import pandas as pd
import numpy as np
import asyncio
import mlflow
from sklearn.metrics import precision_recall_curve, roc_auc_score
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from agents.synthesis_agent import SynthesisAgent
from agents.txn_type import TransactionType

DATA_PATH = Path("data/transaction_features.csv")

MERCHANT_TYPE_TO_TXN_TYPE = {
    "fuel": TransactionType.POS_RETAIL.value,
    "grocery": TransactionType.POS_RETAIL.value,
    "restaurant": TransactionType.POS_RETAIL.value,
    "pharmacy": TransactionType.POS_RETAIL.value,
    "jewelry": TransactionType.POS_RETAIL.value,
    "electronics": TransactionType.ONLINE_MERCHANT.value,
    "fashion": TransactionType.ONLINE_MERCHANT.value,
    "travel": TransactionType.REMITTANCE.value,
    "gaming": TransactionType.QR_CODE.value,
    "crypto": TransactionType.QR_CODE.value,
    "atm": TransactionType.ATM.value,
}


def map_merchant_type_to_txn_type(merchant_type: str) -> str:
    return MERCHANT_TYPE_TO_TXN_TYPE.get(
        str(merchant_type).lower(), TransactionType.QR_CODE.value
    )


def load_data(sample_size: int = 2500):
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

    if sample_size and len(df) > sample_size:
        fraud = df[df["is_fraud"] == 1]
        legit = df[df["is_fraud"] == 0]

        fraud_sample = fraud.sample(n=min(len(fraud), int(sample_size * 0.2)), random_state=42)
        legit_sample = legit.sample(n=min(len(legit), sample_size - len(fraud_sample)), random_state=42)
        df = pd.concat([fraud_sample, legit_sample]).sample(frac=1, random_state=42)

    transactions = []
    labels = []

    for _, row in df.iterrows():
        txn = {
            "user_id": row["user_id"],
            "txn_type": map_merchant_type_to_txn_type(row["merchant_type"]),
            "amount": float(row["amount"]),
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "timestamp": row["timestamp"],
            "merchant_id": row["merchant_id"],
            "user_tenure_years": 3 if row["is_fraud"] == 0 else 0,
            "clean_history": bool(row["is_fraud"] == 0),
            "home_lat": float(row["lat"]),
            "home_lon": float(row["lon"]),
            "device_id": row.get("device_id"),
            "merchant_type": row.get("merchant_type"),
            "transaction_id": row.get("transaction_id"),
        }
        transactions.append(txn)
        labels.append(int(row["is_fraud"]))

    return transactions, np.array(labels)


async def calibrate_thresholds():
    print("Initializing SynthesisAgent...")
    agent = SynthesisAgent()
    print("Agent initialized. Loading data...")
    transactions, labels = load_data(sample_size=100)
    print(f"Loaded {len(transactions)} transactions")

    scores = []
    for i, txn in enumerate(transactions):
        if i % 10 == 0:
            print(f"[Progress] Scoring transaction {i+1}/{len(transactions)}")
        result = await agent.score_transaction(txn)
        scores.append(result.score)

    scores = np.array(scores)
    positive = labels == 1
    negative = labels == 0

    print("[4] Computing metrics...")
    precision, recall, _ = precision_recall_curve(labels, scores)
    roc_auc = roc_auc_score(labels, scores)

    print(f"Sample size: {len(scores)}")
    print(f"ROC AUC: {roc_auc:.4f}")

    thresholds = [0.3, 0.6, 0.8]
    for thresh in thresholds:
        fpr = float(np.mean((scores >= thresh) & negative))
        tpr = float(np.mean((scores >= thresh) & positive))
        print(f"Threshold {thresh}: FPR = {fpr:.4f}, TPR = {tpr:.4f}")

    selected_thresholds = [t for t in thresholds if float(np.mean((scores >= t) & negative)) < 0.02]
    selected_threshold = selected_thresholds[-1] if selected_thresholds else thresholds[0]
    print(f"Selected threshold with FPR < 2%: {selected_threshold}")

    print("[5] Logging to MLflow...")
    mlflow.set_experiment("synthesis_calibration")
    with mlflow.start_run():
        mlflow.log_metric("roc_auc", float(roc_auc))
        for thresh in thresholds:
            mlflow.log_metric(f"false_positive_rate_at_{thresh}", float(np.mean((scores >= thresh) & negative)))
            mlflow.log_metric(f"true_positive_rate_at_{thresh}", float(np.mean((scores >= thresh) & positive)))
        mlflow.log_param("selected_threshold", selected_threshold)
        mlflow.log_param("sample_size", int(len(scores)))

    print("[6] Plotting results...")
    plt.figure()
    plt.plot(recall, precision, marker=".")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Synthesis Precision-Recall Curve")
    plt.savefig("calibration_precision_recall.png")
    mlflow.log_artifact("calibration_precision_recall.png")

    plt.figure()
    plt.hist(scores, bins=25)
    plt.xlabel("Synthesis score")
    plt.ylabel("Count")
    plt.title("Synthesis score distribution")
    plt.savefig("calibration_score_histogram.png")
    mlflow.log_artifact("calibration_score_histogram.png")
    print("[7] Done!")


if __name__ == "__main__":
    asyncio.run(calibrate_thresholds())