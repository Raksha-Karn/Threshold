import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from features.lstm_autoencoder import BehaviourLSTMPredictor
import mlflow
import mlflow.pytorch

INPUT_SIZE = 8
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 50


class TransactionSequenceDataset(Dataset):
    def __init__(self, X_path: str, y_path: str, weights_path: str):
        self.X = torch.tensor(np.load(X_path), dtype=torch.float32)
        self.y = torch.tensor(np.load(y_path), dtype=torch.float32)
        self.weights = torch.tensor(np.load(weights_path), dtype=torch.float32)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index], self.weights[index]
    
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("fraud_anomaly_detection")
def train_model():
    processed_dir = Path("data/processed")
    model_dir = Path("artifacts/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    dataset = TransactionSequenceDataset(
        X_path=processed_dir / "X_train.npy",
        y_path=processed_dir / "y_train.npy",
        weights_path=processed_dir / "train_weights.npy"
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BehaviourLSTMPredictor().to(device=device)
    loss_fn = nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    with mlflow.start_run(run_name="behaviour_lstm"):
        mlflow.log_params({
            "model_type": "BehaviourLSTMPredictor",
            "input_size": INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "loss": "weighted_mse",
        })

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0

            for X_batch, y_batch, weights_batch in dataloader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                weights_batch = weights_batch.to(device)
                
                predictions = model(X_batch)
                per_feature_loss = loss_fn(predictions, y_batch)
                per_sample_loss = per_feature_loss.mean(dim=1)
                weighted_loss = per_sample_loss * weights_batch
                loss = weighted_loss.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * X_batch.size(0)
            avg_loss = total_loss / len(dataset)
            print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.6f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch + 1)
        model_path = model_dir / "behaviour_lstm_state_dict.pt"
        torch.save(model.state_dict(), model_path)

        config = {
            "input_size": INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "model_type": "BehaviourLSTMPredictor",
            "score_type": "mse_reconstruction_error"
        }
        config_path = model_dir / "behaviour_lstm_config.json"

        with open(config_path, "w") as file:
            json.dump(config, file, indent=4)

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(config_path))
        mlflow.log_artifact("artifacts/preprocessors/feature_scaler.joblib")
        mlflow.log_artifact("artifacts/preprocessors/sequence_metadata.json")
        mlflow.log_artifact("data/processed/global_average_sequence.npy")
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name="BehaviourLSTM"
        )

        print("Model saved to: ", model_path)

if __name__ == "__main__":
    train_model()