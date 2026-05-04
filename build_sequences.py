from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import json

SEQUENCE_LENGTH = 30
COLD_START_CONFIDENCE = 0.3
MIN_USER_HISTORY = 5
FEATURE_COLUMNS = [
    "amount_zscore",
    "amount_vs_user_mean",
    "txn_count_in_last_1h",
    "txn_count_in_last_24h",
    "is_new_merchant",
    "hour_of_day",
    "day_of_week",
    "anomaly_score",
]
ID_COLUMNS = [
    "transaction_id",
    "user_id",
    "timestamp"
]
TARGET_COLUMN = "is_fraud"

def load_transactions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_columns = set(ID_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError("Missing required columns in the dataset: ", missing_columns)
    if df["timestamp"].isna().any():
        raise ValueError("Some timestamp values could not be parsed")
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return df

def fit_scaler(df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df[FEATURE_COLUMNS])
    return scaler

def create_global_average_sequence(df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    global_average_vector = pd.DataFrame(
        [df[FEATURE_COLUMNS].mean()],
        columns=FEATURE_COLUMNS
    )
    global_average_vector_scaled = scaler.transform(global_average_vector)
    global_average_sequence = np.repeat(
        global_average_vector_scaled,
        SEQUENCE_LENGTH,
        axis=0
    )
    return global_average_sequence.astype(np.float32)

def build_training_sequences(df: pd.DataFrame, scaler: StandardScaler) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_sequences = []
    y_next_features = []
    confidence_weights = []

    for user_id, group in df.groupby("user_id", sort=False):
        group = group.sort_values("timestamp")
        raw_features = group[FEATURE_COLUMNS]
        scaled_features = scaler.transform(raw_features)
        if len(group) < MIN_USER_HISTORY or len(group) <= SEQUENCE_LENGTH:
            continue
        for i in range(len(group) - SEQUENCE_LENGTH):
            past_sequence = scaled_features[i: i + SEQUENCE_LENGTH]
            next_transaction_features = scaled_features[i + SEQUENCE_LENGTH]
            X_sequences.append(past_sequence)
            y_next_features.append(next_transaction_features)
            confidence_weights.append(1.0)

    X = np.array(X_sequences, dtype=np.float32)
    y = np.array(y_next_features, dtype=np.float32)
    weights = np.array(confidence_weights, dtype=np.float32)
    return X, y, weights

def build_user_sequences_for_inference(df: pd.DataFrame, scaler: StandardScaler, global_average_sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    user_ids = []
    X_sequences = []
    confidence_weights = []
    for user_id, group in df.groupby("user_id", sort=False):
        group = group.sort_values("timestamp")
        raw_features = group[FEATURE_COLUMNS]
        scaled_features = scaler.transform(raw_features)
        user_ids.append(user_id)

        if len(group) < MIN_USER_HISTORY:
            X_sequences.append(global_average_sequence)
            confidence_weights.append(COLD_START_CONFIDENCE)
            continue

        latest_sequence = scaled_features[-SEQUENCE_LENGTH:]
        if len(latest_sequence) < SEQUENCE_LENGTH:
            padding_needed = SEQUENCE_LENGTH - len(latest_sequence)
            padding = np.repeat(
                global_average_sequence[:1],
                padding_needed,
                axis=0
            )
            latest_sequence = np.vstack([padding, latest_sequence])
        X_sequences.append(latest_sequence.astype(np.float32))
        confidence_weights.append(1.0)

    user_ids = np.array(user_ids)
    X = np.array(X_sequences, dtype=np.float32)
    weights = np.array(confidence_weights, dtype=np.float32)
    return user_ids, X, weights

def save_outputs(
        X_train: np.ndarray,
        y_train: np.ndarray,
        train_weights: np.ndarray,
        inference_user_ids: np.ndarray,
        X_inference: np.ndarray,
        inference_weights: np.ndarray,
        scaler: StandardScaler,
        global_average_sequence: np.ndarray,
        output_dir: str
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    np.save(output_path / "X_train.npy", X_train)
    np.save(output_path / "y_train.npy", y_train)
    np.save(output_path / "train_weights.npy", train_weights)
    np.save(output_path / "inference_user_ids.npy", inference_user_ids)
    np.save(output_path / "X_inference.npy", X_inference)
    np.save(output_path / "inference_weights.npy", inference_weights)
    np.save(output_path / "global_average_sequence.npy", global_average_sequence)

    preprocessor_dir = Path("artifacts/preprocessors")
    preprocessor_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, preprocessor_dir / "feature_scaler.joblib")

    metadata = {
        "sequence_length": SEQUENCE_LENGTH,
        "min_user_history": MIN_USER_HISTORY,
        "cold_start_confidence": COLD_START_CONFIDENCE,
        "feature_columns": FEATURE_COLUMNS,
        "input_size": len(FEATURE_COLUMNS),
        "target_type": "next_transaction_feature_reconstruction",
    }
    with open(preprocessor_dir / "sequence_metadata.json", "w") as file:
        json.dump(metadata, file, indent=4)

def main() -> None:
    csv_path = "data/raw/transactions_with_anomalies.csv"
    output_dir = "data/processed"

    df = load_transactions(csv_path=csv_path)
    scaler = fit_scaler(df=df)
    global_average_sequence = create_global_average_sequence(df=df, scaler=scaler)
    X_train, y_train, train_weights = build_training_sequences(
        df=df,
        scaler=scaler
    )
    inference_user_ids, X_inference, inference_weights = (
        build_user_sequences_for_inference(
            df=df,
            scaler=scaler,
            global_average_sequence=global_average_sequence
        )
    )
    save_outputs(
        X_train=X_train,
        y_train=y_train,
        train_weights=train_weights,
        inference_user_ids=inference_user_ids,
        X_inference=X_inference,
        inference_weights=inference_weights,
        scaler=scaler,
        global_average_sequence=global_average_sequence,
        output_dir=output_dir,
    )
    print("Sequence building completed!")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"train_weights shape: {train_weights.shape}")
    print(f"X_inference_latest shape: {X_inference.shape}")
    print(f"inference_weights shape: {inference_weights.shape}")


if __name__ == "__main__":
    main()


