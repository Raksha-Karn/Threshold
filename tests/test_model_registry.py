import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from registry.model_registry import DriftDetector, ModelRouter, ModelVersionManager


class FakeVersion:
    def __init__(self, version):
        self.version = str(version)


class FakeMlflowClient:
    def __init__(self):
        self.models = set()
        self.versions = []
        self.tags = []
        self.aliases = []

    def create_registered_model(self, name):
        if name in self.models:
            raise Exception("already exists")
        self.models.add(name)
        return {"name": name}

    def get_registered_model(self, name):
        return {"name": name}

    def create_model_version(self, name, source, run_id=None, tags=None):
        version = FakeVersion(len(self.versions) + 1)
        self.versions.append(
            {"name": name, "source": source, "run_id": run_id, "tags": tags, "version": version.version}
        )
        return version

    def set_model_version_tag(self, name, version, key, value):
        self.tags.append((name, str(version), key, value))

    def set_registered_model_alias(self, name, alias, version):
        self.aliases.append((name, alias, str(version)))


def test_register_all_v1_sets_tags_and_staging_aliases():
    client = FakeMlflowClient()
    manager = ModelVersionManager(client=client)

    versions = manager.register_all_v1(
        {
            "isolation_forest": "models:/iforest",
            "lstm_behavior": "models:/lstm",
            "velocity_random_forest": "models:/velocity",
            "graph_gcn": "models:/graph",
        }
    )

    assert set(versions) == {
        "isolation_forest",
        "lstm_behavior",
        "velocity_random_forest",
        "graph_gcn",
    }
    assert len(client.versions) == 4
    assert all(item["tags"]["environment"] == "Staging" for item in client.versions)
    assert ("graph_gcn", "staging", "4") in client.aliases


def test_model_router_routes_90_percent_to_production_with_deterministic_keys():
    router = ModelRouter(production_weight=0.90)

    routed = [router.route("isolation_forest", traffic_key=f"txn_{i}") for i in range(1000)]
    production_count = sum(1 for item in routed if item.environment == "Production")

    assert 860 <= production_count <= 940
    assert all(item.model_uri.startswith("models:/isolation_forest@") for item in routed)


def test_model_router_can_force_staging_with_random_fn():
    router = ModelRouter(production_weight=0.90, random_fn=lambda: 0.95)

    routed = router.route("graph_gcn")

    assert routed.environment == "Staging"
    assert routed.model_uri == "models:/graph_gcn@staging"


def test_drift_detector_alerts_on_more_than_5_percent_weekly_drop():
    detector = DriftDetector(drop_threshold=0.05)

    result = detector.evaluate(
        current_metrics={"fraud_catch_rate": 0.84},
        previous_metrics={"fraud_catch_rate": 0.90},
    )

    assert result["alert"] is True
    assert result["trigger_retraining"] is True
    assert round(result["fraud_catch_rate_drop"], 2) == 0.06


def test_drift_detector_ignores_small_drop():
    detector = DriftDetector(drop_threshold=0.05)

    result = detector.evaluate(
        current_metrics={"fraud_catch_rate": 0.86},
        previous_metrics={"fraud_catch_rate": 0.90},
    )

    assert result["alert"] is False
    assert result["trigger_retraining"] is False


def test_log_daily_performance_logs_required_metrics():
    manager = ModelVersionManager(client=FakeMlflowClient())
    run = MagicMock()
    run.info.run_id = "run_123"

    with patch("registry.model_registry.mlflow.set_experiment") as set_experiment:
        with patch("registry.model_registry.mlflow.start_run") as start_run:
            with patch("registry.model_registry.mlflow.log_metric") as log_metric:
                start_run.return_value.__enter__.return_value = run

                run_id = manager.log_daily_performance(
                    {
                        "fraud_catch_rate": 0.91,
                        "false_positive_rate": 0.02,
                        "avg_latency": 240.0,
                    }
                )

    assert run_id == "run_123"
    set_experiment.assert_called_once_with("fraud_model_performance")
    log_metric.assert_any_call("fraud_catch_rate", 0.91)
    log_metric.assert_any_call("false_positive_rate", 0.02)
    log_metric.assert_any_call("avg_latency", 240.0)
