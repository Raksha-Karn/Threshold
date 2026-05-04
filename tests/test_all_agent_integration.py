import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from agents.anomaly_agent import AnomalyAgent
from agents.behaviour_agent import BehaviourAgent
from agents.velocity_rules_agent import VelocityRulesAgent
from agents.graph_agent import GraphAgent


def test_all_agents_return_scores_for_known_fraud():
    anomaly_agent = AnomalyAgent()
    behaviour_agent = BehaviourAgent()
    velocity_agent = VelocityRulesAgent()
    graph_agent = GraphAgent()

    transaction = {
        "user_id": "replace_with_known_fraud_user_id",

        "amount": 5000.0,
        "hour_of_day": 2,
        "day_of_week": 5,
        "amount_zscore": 4.0,
        "user_mean_amount": 200.0,
        "amount_vs_user_mean": 25.0,
        "txn_count_in_last_1h": 5,
        "txn_count_in_last_24h": 20,
        "is_new_merchant": 1,
        "anomaly_score": 0.95,

        "timestamp_epoch": 1770000000.0,
        "merchant_id": "merchant_test",
        "device_id": "device_test",
        "city": "Test City",
        "geo_distance_from_home": 2000.0,
        "card_present_flag": 0,
        "is_international": 1,

        "lat": 40.7128,
        "lon": -74.0060,
        "home_lat": 27.7172,
        "home_lon": 85.3240,
        "timestamp": "2026-05-04T10:00:00+00:00",
    }
    anomaly_score = anomaly_agent.score(transaction)
    behaviour_score = behaviour_agent.score(transaction["user_id"], transaction)
    velocity_score = velocity_agent.score(transaction)
    graph_score = graph_agent.score(transaction["user_id"])

    assert anomaly_score >= 0.0
    assert behaviour_score >= 0.0
    assert velocity_score >= 0.0
    assert graph_score >= 0.0

    assert anomaly_score <= 1.0
    assert velocity_score <= 1.0
    assert graph_score <= 1.0