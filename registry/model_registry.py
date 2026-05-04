from pathlib import Path
import mlflow
import mlflow.pyfunc
import mlflow.artifacts
import mlflow.sklearn

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