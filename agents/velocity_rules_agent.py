import joblib
import pandas as pd


class VelocityRulesAgent:
    def __init__(self, model_path: str = "artifacts/models/velocity_random_forest.pkl"):
        artifact = joblib.load(model_path)

        self.model = artifact["model"]
        self.features = artifact["features"]
        self.threshold = artifact.get("threshold", 0.5)
        self.feature_importances = artifact.get("feature_importances", {})

    def score(self, transaction_features: dict) -> float:
        missing_features = [
            feature for feature in self.features
            if feature not in transaction_features
        ]

        if missing_features:
            raise ValueError(f"Missing velocity features: {missing_features}")

        X = pd.DataFrame([transaction_features])
        X = X.reindex(columns=self.features)
        X = X.fillna(0)

        fraud_probability = self.model.predict_proba(X)[0][1]

        return float(fraud_probability)

    def is_suspicious(self, transaction_features: dict) -> bool:
        return self.score(transaction_features) >= self.threshold