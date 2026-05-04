import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from agents.anomaly_agent import AnomalyAgent


def test_anomaly_agent_returns_score_between_0_and_1(tmp_path):
    features = ["amount", "hour", "merchant_risk"]

    X = np.array([
        [10, 12, 0],
        [15, 13, 0],
        [12, 14, 0],
        [1000, 3, 1],
    ])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(random_state=42, contamination=0.25)
    model.fit(X_scaled)

    raw_scores = -model.decision_function(X_scaled)

    artifact = {
        "model": model,
        "scaler": scaler,
        "features": features,
        "score_min": float(raw_scores.min()),
        "score_max": float(raw_scores.max()),
    }

    model_path = tmp_path / "isolation_forest.pkl"
    joblib.dump(artifact, model_path)

    agent = AnomalyAgent(model_path=str(model_path))

    transaction = {
        "amount": 20,
        "hour": 12,
        "merchant_risk": 0,
    }

    score = agent.score(transaction)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0