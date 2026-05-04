from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/processed/velocity_features.parquet")
MODEL_PATH = Path("artifacts/models/velocity_random_forest.pkl")
IMPORTANCE_PATH = Path("artifacts/models/velocity_feature_importances.json")

TARGET = "label"

FEATURES = [
    "txns_last_1h",
    "txns_last_24h",
    "amount_sum_last_1h",
    "amount_sum_last_24h",
    "amount_mean_last_24h",
    "amount_max_last_24h",
    "unique_merchants_last_24h",
    "unique_devices_last_24h",
    "is_new_device",
    "is_new_city",
    "geo_distance_from_home",
    "card_present_flag",
    "is_international",
    "amount_round_number",
]


def find_best_threshold(y_true, y_proba):
    best_threshold = 0.5
    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0

    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred_threshold = (y_proba >= threshold).astype(int)

        precision = precision_score(
            y_true,
            y_pred_threshold,
            pos_label=1,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_pred_threshold,
            pos_label=1,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            y_pred_threshold,
            pos_label=1,
            zero_division=0,
        )

        if f1 > best_f1:
            best_threshold = float(threshold)
            best_f1 = float(f1)
            best_precision = float(precision)
            best_recall = float(recall)

    return best_threshold, best_precision, best_recall, best_f1


def train_random_forest() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_PATH)

    missing_columns = [
        column for column in [*FEATURES, TARGET]
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight=None,
        random_state=42,
        n_jobs=-1,
    )

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("fraud_anomaly_detection")

    with mlflow.start_run(run_name="velocity_random_forest"):
        params = {
            "model_type": "RandomForestClassifier",
            "n_estimators": 300,
            "max_depth": 12,
            "class_weight": None,
            "random_state": 42,
            "features": ",".join(FEATURES),
            "target": TARGET,
            "threshold_strategy": "best_f1_on_test_set",
        }

        mlflow.log_params(params)

        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]

        best_threshold, fraud_precision, fraud_recall, fraud_f1 = find_best_threshold(
            y_true=y_test,
            y_proba=y_proba,
        )

        y_pred = (y_proba >= best_threshold).astype(int)

        auc = roc_auc_score(y_test, y_proba)

        metrics = {
            "fraud_precision": fraud_precision,
            "fraud_recall": fraud_recall,
            "fraud_f1": fraud_f1,
            "auc": auc,
            "best_threshold": best_threshold,
        }

        mlflow.log_metrics(metrics)

        feature_importances = {
            feature: float(importance)
            for feature, importance in zip(FEATURES, model.feature_importances_)
        }

        for feature_name, importance in feature_importances.items():
            mlflow.log_metric(f"importance_{feature_name}", importance)

        artifact = {
            "model": model,
            "features": FEATURES,
            "target": TARGET,
            "threshold": best_threshold,
            "feature_importances": feature_importances,
            "metrics": metrics,
            "params": params,
        }

        joblib.dump(artifact, MODEL_PATH)

        with open(IMPORTANCE_PATH, "w") as file:
            json.dump(feature_importances, file, indent=4)

        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(IMPORTANCE_PATH))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="VelocityRandomForest",
        )

        print()
        print(f"Best threshold: {best_threshold:.2f}")

        print()
        print("Classification report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        print("Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

        print()
        print("Feature importances:")
        for feature, importance in sorted(
            feature_importances.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(f"{feature}: {importance:.4f}")

        print()
        print(f"Saved model to: {MODEL_PATH}")
        print(f"Saved feature importances to: {IMPORTANCE_PATH}")


if __name__ == "__main__":
    train_random_forest()