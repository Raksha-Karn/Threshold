import joblib
import pandas as pd
import math


class AnomalyAgent:
    def __init__(self, model_path: str = "models/isolation_forest.pkl"):
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.scaler = artifact["scaler"]
        self.features = artifact["features"]
        self.score_min = artifact["score_min"]
        self.score_max = artifact["score_max"]

    def score(self, transaction: dict) -> float:
        X = pd.DataFrame([transaction])
        X = X.reindex(columns=self.features)
        X = X.fillna(0)
        X_scaled = self.scaler.transform(X)
        raw_score = -self.model.decision_function(X_scaled)[0]
     
        if self.score_max == self.score_min:
            return 0.0

        normalized_score = (raw_score - self.score_min) / (
            self.score_max - self.score_min
        )

        normalized_score = max(0.0, min(1.0, normalized_score))

        return float(normalized_score)
