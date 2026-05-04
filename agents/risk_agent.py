class RiskAgent:
    def __init__(
            self,
            anomaly_agent,
            behaviour_agent,
            cold_start_threshold: float = 0.85,
            normal_threshold: float = 0.70,
            behaviour_weight: float = 0.6,
            anomaly_weight: float = 0.4
    ):
        self.anomaly_agent = anomaly_agent
        self.behaviour_agent = behaviour_agent
        self.cold_start_threshold = cold_start_threshold
        self.normal_threshold = normal_threshold
        self.behaviour_weight = behaviour_weight
        self.anomaly_weight = anomaly_weight

    def score(self, user_id: str, transaction: dict) -> dict:
        anomaly_score = self.anomaly_agent.score(transaction)
        if self.behaviour_agent.is_cold_start(user_id):
            final_score = anomaly_score
            threshold = self.cold_start_threshold
            decision_reason = "cold_start_isolation_forest_only"
            behaviour_score = None
        else:
            behaviour_score = self.behaviour_agent.score(user_id, transaction)
            final_score = (
                self.behaviour_weight * behaviour_score
                + self.anomaly_weight * anomaly_score
            )
            threshold = self.normal_threshold
            decision_reason = "combined_lstm_and_isolation_forest"
        return {
            "user_id": user_id,
            "anomaly_score": float(anomaly_score),
            "behaviour_score": None if behaviour_score is None else float(behaviour_score),
            "threshold": float(threshold),
            "final_score": float(final_score),
            "is_suspicious": bool(final_score >= threshold),
            "reason": decision_reason
        }