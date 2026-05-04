import json
import time
from typing import Any
import joblib
import pandas as pd
import redis


class VelocityRulesAgent:
    def __init__(
        self,
        model_path: str = "artifacts/models/velocity_random_forest.pkl",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        history_ttl_seconds: int = 60 * 60 * 48,
    ):
        artifact = joblib.load(model_path)

        self.model = artifact["model"]
        self.features = artifact["features"]
        self.threshold = artifact.get("threshold", 0.5)
        self.feature_importances = artifact.get("feature_importances", {})

        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True,
        )

        self.history_ttl_seconds = history_ttl_seconds

    def _user_key(self, user_id: str) -> str:
        return f"velocity:{user_id}:transactions"

    def _get_recent_transactions(
        self,
        user_id: str,
        current_timestamp: float,
        window_seconds: int,
    ) -> list[dict[str, Any]]:
        key = self._user_key(user_id)
        window_start = current_timestamp - window_seconds

        raw_transactions = self.redis_client.zrangebyscore(
            key,
            min=window_start,
            max=current_timestamp,
        )

        transactions = []

        for raw_transaction in raw_transactions:
            try:
                transactions.append(json.loads(raw_transaction))
            except json.JSONDecodeError:
                continue

        return transactions

    def _store_transaction(
        self,
        user_id: str,
        transaction: dict,
        current_timestamp: float,
    ) -> None:
        key = self._user_key(user_id)

        redis_record = {
            "timestamp": current_timestamp,
            "amount": float(transaction.get("amount", 0.0)),
            "merchant_id": str(transaction.get("merchant_id", "")),
            "device_id": str(transaction.get("device_id", "")),
            "city": str(transaction.get("city", "")),
        }

        self.redis_client.zadd(
            key,
            {json.dumps(redis_record): current_timestamp},
        )

        oldest_allowed_timestamp = current_timestamp - self.history_ttl_seconds

        self.redis_client.zremrangebyscore(
            key,
            min=0,
            max=oldest_allowed_timestamp,
        )

        self.redis_client.expire(key, self.history_ttl_seconds)

    def _build_realtime_features(self, transaction: dict) -> dict:
        user_id = str(transaction["user_id"])

        current_timestamp = float(
            transaction.get("timestamp_epoch", time.time())
        )

        recent_1h = self._get_recent_transactions(
            user_id=user_id,
            current_timestamp=current_timestamp,
            window_seconds=60 * 60,
        )

        recent_24h = self._get_recent_transactions(
            user_id=user_id,
            current_timestamp=current_timestamp,
            window_seconds=60 * 60 * 24,
        )

        amounts_1h = [
            float(txn.get("amount", 0.0))
            for txn in recent_1h
        ]

        amounts_24h = [
            float(txn.get("amount", 0.0))
            for txn in recent_24h
        ]

        merchant_ids_24h = {
            str(txn.get("merchant_id", ""))
            for txn in recent_24h
            if txn.get("merchant_id") is not None
        }

        device_ids_24h = {
            str(txn.get("device_id", ""))
            for txn in recent_24h
            if txn.get("device_id") is not None
        }

        previous_device_ids = {
            str(txn.get("device_id", ""))
            for txn in recent_24h
            if txn.get("device_id") is not None
        }

        previous_cities = {
            str(txn.get("city", ""))
            for txn in recent_24h
            if txn.get("city") is not None
        }

        current_device_id = str(transaction.get("device_id", ""))
        current_city = str(transaction.get("city", ""))

        amount = float(transaction.get("amount", 0.0))

        amount_mean_last_24h = (
            sum(amounts_24h) / len(amounts_24h)
            if amounts_24h
            else 0.0
        )

        amount_max_last_24h = (
            max(amounts_24h)
            if amounts_24h
            else 0.0
        )

        features = {
            "txns_last_1h": len(recent_1h),
            "txns_last_24h": len(recent_24h),
            "amount_sum_last_1h": sum(amounts_1h),
            "amount_sum_last_24h": sum(amounts_24h),
            "amount_mean_last_24h": amount_mean_last_24h,
            "amount_max_last_24h": amount_max_last_24h,
            "unique_merchants_last_24h": len(merchant_ids_24h),
            "unique_devices_last_24h": len(device_ids_24h),
            "is_new_device": int(current_device_id not in previous_device_ids),
            "is_new_city": int(current_city not in previous_cities),
            "geo_distance_from_home": float(
                transaction.get("geo_distance_from_home", 0.0)
            ),
            "card_present_flag": int(
                transaction.get("card_present_flag", 0)
            ),
            "is_international": int(
                transaction.get("is_international", 0)
            ),
            "amount_round_number": int(
                (amount % 10 == 0)
                or (amount % 50 == 0)
                or (amount % 100 == 0)
            ),
        }

        return features

    def score(self, transaction: dict, update_redis: bool = True) -> float:
        if "user_id" not in transaction:
            raise ValueError("Transaction must include user_id.")

        features = self._build_realtime_features(transaction)

        missing_features = [
            feature for feature in self.features
            if feature not in features
        ]

        if missing_features:
            raise ValueError(f"Missing velocity features: {missing_features}")

        X = pd.DataFrame([features])
        X = X.reindex(columns=self.features)
        X = X.fillna(0)

        fraud_probability = self.model.predict_proba(X)[0][1]

        if update_redis:
            current_timestamp = float(
                transaction.get("timestamp_epoch", time.time())
            )

            self._store_transaction(
                user_id=str(transaction["user_id"]),
                transaction=transaction,
                current_timestamp=current_timestamp,
            )

        return float(fraud_probability)

    def is_suspicious(self, transaction: dict) -> bool:
        return self.score(transaction) >= self.threshold