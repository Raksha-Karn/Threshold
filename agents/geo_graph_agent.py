import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from agents.geo_agent import GeoAgent
from agents.graph_agent import GraphAgent


class GeoGraphAgent:
    def __init__(
        self,
        geo_agent: GeoAgent | None = None,
        graph_agent: GraphAgent | None = None,
        geo_weight: float = 0.4,
        graph_weight: float = 0.6,
    ):
        self.geo_agent = geo_agent or GeoAgent()
        self.graph_agent = graph_agent or GraphAgent()
        self.geo_weight = geo_weight
        self.graph_weight = graph_weight

    def score(self, transaction: dict) -> float:
        if "user_id" not in transaction:
            raise ValueError("Transaction must include user_id.")

        user_id = str(transaction["user_id"])

        geo_score = self.geo_agent.score(transaction)
        graph_score = self.graph_agent.score(user_id)

        final_score = (
            self.geo_weight * geo_score
            + self.graph_weight * graph_score
        )

        return float(final_score)

    def explain(self, transaction: dict) -> dict:
        user_id = str(transaction["user_id"])

        geo_score = self.geo_agent.score(transaction, update_history=False)
        graph_score = self.graph_agent.score(user_id)

        final_score = (
            self.geo_weight * geo_score
            + self.graph_weight * graph_score
        )

        return {
            "user_id": user_id,
            "geo_score": float(geo_score),
            "graph_score": float(graph_score),
            "final_score": float(final_score),
            "geo_weight": self.geo_weight,
            "graph_weight": self.graph_weight,
        }

    def is_suspicious(self, transaction: dict, threshold: float = 0.5) -> bool:
        return self.score(transaction) >= threshold