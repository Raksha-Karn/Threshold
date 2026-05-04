import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from agents.geo_agent import GeoAgent


def test_geo_agent_returns_score_between_0_and_1():
    agent = GeoAgent()

    transaction = {
        "user_id": "user_1",
        "lat": 27.7172,
        "lon": 85.3240,
        "home_lat": 27.7172,
        "home_lon": 85.3240,
        "timestamp": "2026-05-04T10:00:00+00:00",
    }

    score = agent.score(transaction)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_geo_agent_flags_impossible_travel():
    agent = GeoAgent()

    first_transaction = {
        "user_id": "user_1",
        "lat": 27.7172,
        "lon": 85.3240,
        "home_lat": 27.7172,
        "home_lon": 85.3240,
        "timestamp": "2026-05-04T10:00:00+00:00",
    }

    second_transaction = {
        "user_id": "user_1",
        "lat": 40.7128,
        "lon": -74.0060,
        "home_lat": 27.7172,
        "home_lon": 85.3240,
        "timestamp": "2026-05-04T11:00:00+00:00",
    }

    agent.score(first_transaction)
    score = agent.score(second_transaction)

    assert score >= 0.5