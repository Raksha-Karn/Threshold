import json
import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from agents.behaviour_agent import BehaviourAgent
from features.lstm_autoencoder import BehaviourLSTMPredictor


def test_behaviour_agent_returns_float_score(tmp_path):
    feature_columns = [
    "amount_zscore",
    "amount_vs_user_mean",
    "txn_count_in_last_1h",
    "txn_count_in_last_24h",
    "is_new_merchant",
    "hour_of_day",
    "day_of_week",
    "anomaly_score",
    ]

    sequence_length = 30

    metadata = {
        "feature_columns": feature_columns,
        "sequence_length": sequence_length,
        "min_user_history": 5,
        "cold_start_confidence": 0.3,
    }

    metadata_path = tmp_path / "sequence_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    config = {
        "input_size": 8,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.3,
    }

    config_path = tmp_path / "behaviour_lstm_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

    scaler = StandardScaler()
    fake_training_data = np.random.rand(100, 8)
    scaler.fit(fake_training_data)

    scaler_path = tmp_path / "feature_scaler.joblib"
    joblib.dump(scaler, scaler_path)

    global_average_sequence = np.zeros((sequence_length, 8), dtype=np.float32)
    global_average_sequence_path = tmp_path / "global_average_sequence.npy"
    np.save(global_average_sequence_path, global_average_sequence)

    model = BehaviourLSTMPredictor(
        input_size=8,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
    )

    model_path = tmp_path / "behaviour_lstm_state_dict.pt"
    torch.save(model.state_dict(), model_path)

    agent = BehaviourAgent(
        model_path=str(model_path),
        config_path=str(config_path),
        scaler_path=str(scaler_path),
        metadata_path=str(metadata_path),
        global_average_sequence_path=str(global_average_sequence_path),
    )

    transaction = {
        column: 1.0 for column in feature_columns
    }

    score = agent.score("user_123", transaction)

    assert isinstance(score, float)
    assert score >= 0.0