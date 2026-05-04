from agents.risk_agent import RiskAgent


class FakeAnomalyAgent:
    def score(self, transaction):
        return 0.9


class FakeBehaviourAgent:
    def __init__(self, cold_start):
        self.cold_start = cold_start

    def is_cold_start(self, user_id):
        return self.cold_start

    def score(self, user_id, transaction):
        return 0.2


def test_risk_agent_uses_isolation_forest_only_for_cold_start():
    risk_agent = RiskAgent(
        anomaly_agent=FakeAnomalyAgent(),
        behaviour_agent=FakeBehaviourAgent(cold_start=True),
        cold_start_threshold=0.85,
    )

    result = risk_agent.score("new_user", {"amount": 100})

    assert result["behaviour_score"] is None
    assert result["final_score"] == 0.9
    assert result["is_suspicious"] is True
    assert result["reason"] == "cold_start_isolation_forest_only"


def test_risk_agent_combines_scores_for_existing_user():
    risk_agent = RiskAgent(
        anomaly_agent=FakeAnomalyAgent(),
        behaviour_agent=FakeBehaviourAgent(cold_start=False),
        behaviour_weight=0.6,
        anomaly_weight=0.4,
    )

    result = risk_agent.score("old_user", {"amount": 100})

    expected_score = 0.6 * 0.2 + 0.4 * 0.9

    assert result["behaviour_score"] == 0.2
    assert result["final_score"] == expected_score
    assert result["reason"] == "combined_lstm_and_isolation_forest"