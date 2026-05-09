from pathlib import Path
import hashlib
import random
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import mlflow
import mlflow.pyfunc
import mlflow.artifacts
import mlflow.sklearn
from mlflow.tracking import MlflowClient


MODEL_VERSION_SOURCES = {
    "isolation_forest": "models/isolation_forest.pkl",
    "lstm_behavior": "artifacts/models/behaviour_lstm_state_dict.pt",
    "velocity_random_forest": "artifacts/models/velocity_random_forest.pkl",
    "graph_gcn": "artifacts/models/fraud_gcn_state_dict.pt",
}


@dataclass(frozen=True)
class RoutedModel:
    model_name: str
    environment: str
    model_uri: str

class ModelRegistry:
    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        model_alias: str = "champion",
        download_dir: str = "artifacts/mlflow_downloads",
    ):
        self.tracking_uri = tracking_uri
        self.model_alias = model_alias
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        mlflow.set_tracking_uri(self.tracking_uri)

    def load_sklearn_model(self, registered_model_name: str):
        model_uri = f"models:/{registered_model_name}@{self.model_alias}"
        return mlflow.sklearn.load_model(model_uri)

    def load_pyfunc_model(self, registered_model_name: str):
        model_uri = f"models:/{registered_model_name}@{self.model_alias}"
        return mlflow.pyfunc.load_model(model_uri)

    def download_artifact(self, run_id: str, artifact_path: str):
        return mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
            dst_path=str(self.download_dir),
        )


class ModelVersionManager:
    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        client: Optional[MlflowClient] = None,
    ):
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = client or MlflowClient(tracking_uri=self.tracking_uri)

    def ensure_registered_model(self, model_name: str):
        try:
            return self.client.create_registered_model(model_name)
        except Exception:
            return self.client.get_registered_model(model_name)

    def register_model_version(
        self,
        model_name: str,
        source_uri: str,
        version_tag: str,
        environment: str = "Staging",
        run_id: Optional[str] = None,
    ):
        self.ensure_registered_model(model_name)
        version = self.client.create_model_version(
            name=model_name,
            source=source_uri,
            run_id=run_id,
            tags={
                "version_tag": version_tag,
                "environment": environment,
            },
        )
        self.set_environment(model_name, version.version, environment)
        return version

    def register_all_v1(self, source_map: Optional[Dict[str, str]] = None) -> dict:
        source_map = source_map or MODEL_VERSION_SOURCES
        versions = {}
        for model_name, source_uri in source_map.items():
            versions[model_name] = self.register_model_version(
                model_name=model_name,
                source_uri=source_uri,
                version_tag=f"{model_name}/v1",
                environment="Staging",
            )
        return versions

    def set_environment(self, model_name: str, version: str, environment: str):
        self.client.set_model_version_tag(model_name, version, "environment", environment)
        alias = environment.lower()
        if hasattr(self.client, "set_registered_model_alias"):
            self.client.set_registered_model_alias(model_name, alias, version)
        return {"model_name": model_name, "version": str(version), "environment": environment}

    def archive_version(self, model_name: str, version: str):
        return self.set_environment(model_name, version, "Archived")

    def log_daily_performance(
        self,
        metrics: Dict[str, float],
        experiment_name: str = "fraud_model_performance",
    ) -> str:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="daily_model_performance") as run:
            for name, value in metrics.items():
                mlflow.log_metric(name, float(value))
            return run.info.run_id


class ModelRouter:
    def __init__(
        self,
        production_weight: float = 0.90,
        random_fn: Optional[Callable[[], float]] = None,
    ):
        if not 0.0 <= production_weight <= 1.0:
            raise ValueError("production_weight must be between 0 and 1")
        self.production_weight = production_weight
        self.random_fn = random_fn or random.random

    def choose_environment(self, traffic_key: Optional[str] = None) -> str:
        if traffic_key is None:
            bucket = self.random_fn()
        else:
            digest = hashlib.sha256(str(traffic_key).encode("utf-8")).hexdigest()
            bucket = int(digest[:8], 16) / 0xFFFFFFFF
        return "Production" if bucket < self.production_weight else "Staging"

    def route(self, model_name: str, traffic_key: Optional[str] = None) -> RoutedModel:
        environment = self.choose_environment(traffic_key)
        alias = environment.lower()
        return RoutedModel(
            model_name=model_name,
            environment=environment,
            model_uri=f"models:/{model_name}@{alias}",
        )


class DriftDetector:
    def __init__(self, drop_threshold: float = 0.05):
        self.drop_threshold = drop_threshold

    def evaluate(
        self,
        current_metrics: Dict[str, float],
        previous_metrics: Dict[str, float],
    ) -> Dict[str, object]:
        previous = float(previous_metrics.get("fraud_catch_rate", 0.0))
        current = float(current_metrics.get("fraud_catch_rate", 0.0))
        drop = previous - current
        should_alert = previous > 0 and drop > self.drop_threshold
        return {
            "alert": should_alert,
            "trigger_retraining": should_alert,
            "fraud_catch_rate_drop": drop,
            "reason": "fraud_catch_rate_drop" if should_alert else "within_threshold",
        }
