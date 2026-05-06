import asyncio
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

from agents.behaviour_agent import BehaviourAgent
from agents.graph_agent import GraphAgent
from agents.velocity_rules_agent import VelocityRulesAgent
from agents.anomaly_agent import AnomalyAgent
from agents.geo_agent import GeoAgent
from agents.txn_type import TransactionType


@dataclass
class SynthesisResult:
    score: float
    verdict: str
    agent_scores: Dict[str, float]
    weights_used: Dict[str, float]
    confidence: float


class SynthesisAgent:
    def __init__(
        self,
        weights_config_path: str = "weights_config.yaml",
        merchant_whitelist_path: str = "merchant_whitelist.yaml",
    ):
        self.weights_config_path = Path(weights_config_path)
        self.merchant_whitelist_path = Path(merchant_whitelist_path)
        self.weights_config = self._load_weights_config()
        self.merchant_whitelist = self._load_merchant_whitelist()

        self.behaviour_agent = BehaviourAgent()
        self.graph_agent = GraphAgent()
        self.velocity_agent = VelocityRulesAgent()
        self.anomaly_agent = AnomalyAgent()
        self.geo_agent = GeoAgent()

        self.executor = ThreadPoolExecutor(max_workers=5)

    def _load_weights_config(self) -> Dict[str, Dict[str, float]]:
        with open(self.weights_config_path, "r") as f:
            config = yaml.safe_load(f)
        return config.get("txn_weights", {})

    def _load_merchant_whitelist(self) -> set:
        try:
            with open(self.merchant_whitelist_path, "r") as f:
                config = yaml.safe_load(f)
            return set(config.get("merchant_whitelist", []))
        except FileNotFoundError:
            return set()

    async def _score_in_executor(self, func, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, func, *args)

    def _get_weights_for_txn_type(self, txn_type: str) -> Dict[str, float]:
        return self.weights_config.get(txn_type, {})

    def _user_trust_adjustment(self, transaction: Dict[str, Any]) -> float:
        if (
            transaction.get("user_tenure_years", 0) > 2
            and transaction.get("clean_history", False)
        ):
            return -0.15
        return 0.0

    def _merchant_whitelist_adjustment(self, transaction: Dict[str, Any]) -> float:
        merchant_id = transaction.get("merchant_id")
        if merchant_id in self.merchant_whitelist:
            return -0.1
        return 0.0

    def _time_of_day_prior(self, timestamp) -> float:
        if timestamp:
            from datetime import datetime
            # Handle both string and datetime objects
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    return 0.0
            
            if hasattr(timestamp, 'hour'):
                hour = timestamp.hour
                if 2 <= hour < 3:
                    return 0.05
                elif 9 <= hour < 18:
                    return -0.05
        return 0.0

    def _compute_verdict(self, score: float) -> str:
        if score < 0.3:
            return "APPROVE"
        elif score < 0.6:
            return "REVIEW"
        elif score < 0.8:
            return "OTP_REQUIRED"
        else:
            return "BLOCK"

    async def score_transaction(self, transaction: Dict[str, Any]) -> SynthesisResult:
        txn_type = transaction.get("txn_type")
        if not txn_type:
            raise ValueError("Transaction must include 'txn_type'")

        user_id = transaction.get("user_id")
        if not user_id:
            raise ValueError("Transaction must include 'user_id'")

        weights = self._get_weights_for_txn_type(txn_type)

        tasks = [
            self._score_in_executor(self.behaviour_agent.score, user_id, transaction),
            self._score_in_executor(self.graph_agent.score, user_id),
            self._score_in_executor(self.velocity_agent.score, transaction),
            self._score_in_executor(self.anomaly_agent.score, transaction),
            self._score_in_executor(self.geo_agent.score, transaction),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        agent_names = ["behaviour_agent", "graph_agent", "velocity_agent", "anomaly_agent", "geo_agent"]
        agent_scores = {}
        successful_scores = {}

        for name, result in zip(agent_names, results):
            if isinstance(result, Exception):
                continue
            agent_scores[name] = result
            if name in weights:
                successful_scores[name] = result

        valid_weights = {name: weights.get(name, 0) for name in successful_scores}
        weight_sum = sum(valid_weights.values())
        if weight_sum == 0:
            weights_used = {name: 1.0 / len(successful_scores) for name in successful_scores}
        else:
            weights_used = {name: w / weight_sum for name, w in valid_weights.items()}

        raw_score = sum(weights_used[name] * score for name, score in successful_scores.items())

        trust_adjustment = self._user_trust_adjustment(transaction)
        merchant_adjustment = self._merchant_whitelist_adjustment(transaction)
        time_prior = self._time_of_day_prior(transaction.get("timestamp"))

        final_score = max(0.0, min(1.0, raw_score + trust_adjustment + merchant_adjustment + time_prior))

        confidence = len(successful_scores) / len(agent_names)

        verdict = self._compute_verdict(final_score)

        return SynthesisResult(
            score=final_score,
            verdict=verdict,
            agent_scores=agent_scores,
            weights_used=weights_used,
            confidence=confidence,
        )