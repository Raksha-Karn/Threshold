import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from agents.graph_agent import GraphAgent


def test_graph_agent_returns_probability_between_0_and_1():
    agent = GraphAgent()

    with open("data/processed/graph_metadata.json", "r") as file:
        metadata = json.load(file)

    user_id = metadata["user_ids"][0]

    score = agent.score(user_id)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_graph_agent_unknown_user_returns_zero():
    agent = GraphAgent()

    score = agent.score("unknown_user_id")

    assert score == 0.0