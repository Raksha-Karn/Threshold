from pathlib import Path
import mlflow


ARTIFACTS = [
    "artifacts/models/velocity_random_forest.pkl",
    "artifacts/models/velocity_feature_importances.json",
    "artifacts/models/behaviour_lstm_state_dict.pt",
    "artifacts/models/behaviour_lstm_config.json",
    "artifacts/models/fraud_gcn_state_dict.pt",
    "artifacts/models/fraud_gcn_config.json",
    "data/processed/graph_metadata.json",
]

def main():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("fraud_anomaly_detection")

    with mlflow.start_run(run_name="all_agent_artifacts"):
        for artifact_path in ARTIFACTS:
            path = Path(artifact_path)

            if path.exists():
                mlflow.log_artifact(str(path))
                print(f"Logged: {path}")
            else:
                print(f"Missing, skipped: {path}")


if __name__ == "__main__":
    main()