from pathlib import Path
import json
import numpy as np
import torch
from collections import defaultdict, deque
import joblib
from torch import nn
import pandas as pd
from lstm_autoencoder import BehaviourLSTMPredictor


class BehaviourAgent:
    def __init__(
            self,
            model_path: str = "artifacts/models/behaviour_lstm_state_dict.pt",
            config_path: str = "artifacts/models/behaviour_lstm_config.json",
            scaler_path: str = "artifacts/preprocessors/feature_scaler.joblib",
            metadata_path: str = "artifacts/preprocessors/sequence_metadata.json",
            global_average_sequence_path: str = "data/processed/global_average_sequence.npy"
        ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(config_path, "r") as file:
            self.config = json.load(file)
        with open(metadata_path, "r") as file:
            self.metadata = json.load(file)
        self.feature_columns = self.metadata["feature_columns"]
        self.sequence_length = self.metadata["sequence_length"]
        self.min_user_history = self.metadata["min_user_history"]
        self.cold_start_confidence = self.metadata["cold_start_confidence"]
        self.scaler = joblib.load(scaler_path)
        self.global_average_sequence = np.load(global_average_sequence_path).astype(np.float32)
        self.model = BehaviourLSTMPredictor(
            input_size=self.config["input_size"],
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"]
        ).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict=state_dict)
        self.model.eval()
        self.user_histories = defaultdict(lambda: deque(maxlen=self.sequence_length))

    def is_cold_start(self, user_id: str) -> bool:
        return len(self.user_histories[user_id]) < self.min_user_history

    def _transaction_to_scaled_features(self, transaction: dict) -> np.ndarray:
        missing_features = [
            column for column in self.feature_columns if column not in transaction
        ]
        if missing_features:
            raise ValueError(f"Transaction has {len(missing_features)} missing features: ", missing_features)
        feature_df = pd.DataFrame(
            [[transaction[column] for column in self.feature_columns]],
            columns=self.feature_columns
        )
        scaled_features = self.scaler.transform(feature_df)[0]
        return scaled_features.astype(np.float32)
    
    def _build_sequence_for_user(self, user_id: str) -> tuple[np.ndarray, float]:
        history = list(self.user_histories[user_id])
        if len(history) < self.min_user_history:
            return self.global_average_sequence.copy(), self.cold_start_confidence
        latest_sequence = np.array(history[-self.sequence_length:], dtype=np.float32)
        if len(latest_sequence) < self.sequence_length:
            padding_needed = self.sequence_length - len(latest_sequence)
            padding = np.repeat(
                self.global_average_sequence[:1],
                padding_needed,
                axis=0
            )
            latest_sequence = np.vstack([padding, latest_sequence])
        return latest_sequence.astype(np.float32), 1.0
    
    def score(self, user_id: str, transaction: dict) -> float:
        scaled_current_features = self._transaction_to_scaled_features(transaction=transaction)
        sequence, confidence = self._build_sequence_for_user(user_id=user_id)
        X = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            predicted_next_features = self.model(X)
            y_true = torch.tensor(scaled_current_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            mse_error = torch.mean((predicted_next_features - y_true) ** 2).item()
            final_score = mse_error * confidence
            self.user_histories[user_id].append(scaled_current_features)

            return float(final_score)