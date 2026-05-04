from pathlib import Path
import pandas as pd
import numpy as np

INPUT_PATH = Path("data/raw/transactions_with_anomalies.csv")
OUTPUT_PATH = Path("data/processed/velocity_features.parquet")

MIN_HOME_TRANSACTIONS = 5
INTERNATIONAL_DISTANCE_KM_THRESHOLD = 1000.0
REQUIRED_INPUT_COLUMNS = [
    "timestamp",
    "transaction_id",
    "user_id",
    "amount",
    "city",
    "lat",
    "lon",
    "merchant_type",
    "merchant_id",
    "device_id",
    "is_fraud",
    "txn_count_in_last_1h",
    "txn_count_in_last_24h",
]
FINAL_FEATURE_COLUMNS = [
    "txns_last_1h",
    "txns_last_24h",
    "amount_sum_last_1h",
    "amount_sum_last_24h",
    "amount_mean_last_24h",
    "amount_max_last_24h",
    "unique_merchants_last_24h",
    "unique_devices_last_24h",
    "is_new_device",
    "is_new_city",
    "geo_distance_from_home",
    "is_international",
    "amount_round_number",
]

def haversine_distance_km(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:
    earth_radius_km = 6371.0

    lat1_rad = np.radians(lat1.astype(float))
    lon1_rad = np.radians(lon1.astype(float))
    lat2_rad = np.radians(lat2.astype(float))
    lon2_rad = np.radians(lon2.astype(float))

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return earth_radius_km * c

def infer_user_home_locations(df: pd.DataFrame) -> pd.DataFrame:
    df_sorted = df.sort_values(["user_id", "timestamp"]).copy()

    first_transactions = (
        df_sorted
        .groupby("user_id", group_keys=False)
        .head(MIN_HOME_TRANSACTIONS)
    )
    home_locations = (
        first_transactions
        .groupby("user_id")
        .agg(
            home_lat=("lat", "median"),
            home_lon=("lon", "median"),
            home_city=("city", lambda values: values.mode().iloc[0] if not values.mode().empty else values.iloc[0]),
        )
        .reset_index()
    )
    return home_locations

def add_geo_distance_from_home(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    home_locations = infer_user_home_locations(df)
    df = df.merge(home_locations, on="user_id", how="left")
    global_home_lat = df["lat"].median()
    global_home_lon = df["lon"].median()

    df["home_lat"] = df["home_lat"].fillna(global_home_lat)
    df["home_lon"] = df["home_lon"].fillna(global_home_lon)
    df["geo_distance_from_home"] = haversine_distance_km(
        lat1=df["home_lat"],
        lon1=df["home_lon"],
        lat2=df["lat"],
        lon2=df["lon"],
    )
    df["geo_distance_from_home"] = df["geo_distance_from_home"].fillna(0.0)
    return df

def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["txns_last_1h"] = (
    df["txn_count_in_last_1h"].fillna(0).astype(float) - 1
    ).clip(lower=0)

    df["txns_last_24h"] = (
        df["txn_count_in_last_24h"].fillna(0).astype(float) - 1
    ).clip(lower=0)
    return df

def add_rich_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["user_id", "timestamp"])

    result_frames = []

    for user_id, user_df in df.groupby("user_id", group_keys=False):
        user_df = user_df.sort_values("timestamp").copy()
        user_df = user_df.set_index("timestamp")

        user_df["amount_sum_last_1h"] = (
            user_df["amount"]
            .rolling("1h", closed="left")
            .sum()
            .fillna(0)
        )

        user_df["amount_sum_last_24h"] = (
            user_df["amount"]
            .rolling("24h", closed="left")
            .sum()
            .fillna(0)
        )

        user_df["amount_mean_last_24h"] = (
            user_df["amount"]
            .rolling("24h", closed="left")
            .mean()
            .fillna(0)
        )

        user_df["amount_max_last_24h"] = (
            user_df["amount"]
            .rolling("24h", closed="left")
            .max()
            .fillna(0)
        )

        timestamps = user_df.index.to_list()
        merchant_ids = user_df["merchant_id"].astype(str).to_list()
        device_ids = user_df["device_id"].astype(str).to_list()

        unique_merchants_last_24h = []
        unique_devices_last_24h = []

        for current_index, current_time in enumerate(timestamps):
            window_start = current_time - pd.Timedelta(hours=24)

            previous_indices = [
                index
                for index in range(current_index)
                if timestamps[index] >= window_start
            ]

            merchants_in_window = {
                merchant_ids[index]
                for index in previous_indices
            }

            devices_in_window = {
                device_ids[index]
                for index in previous_indices
            }

            unique_merchants_last_24h.append(len(merchants_in_window))
            unique_devices_last_24h.append(len(devices_in_window))

        user_df["unique_merchants_last_24h"] = unique_merchants_last_24h
        user_df["unique_devices_last_24h"] = unique_devices_last_24h

        seen_devices = set()
        is_new_device_values = []

        for device_id in user_df["device_id"].astype(str):
            is_new_device_values.append(int(device_id not in seen_devices))
            seen_devices.add(device_id)

        user_df["is_new_device"] = is_new_device_values

        seen_cities = set()
        is_new_city_values = []

        for city in user_df["city"].astype(str):
            is_new_city_values.append(int(city not in seen_cities))
            seen_cities.add(city)

        user_df["is_new_city"] = is_new_city_values

        user_df = user_df.reset_index()
        result_frames.append(user_df)

    return pd.concat(result_frames, ignore_index=True)

def add_amount_round_number(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    amount = df["amount"].fillna(0)
    df["amount_round_number"] = (
        (amount % 10 == 0)
        | (amount % 50 == 0)
        | (amount % 100 == 0)
    ).astype(int)
    return df

def add_is_international_proxy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_international"] = (
        df["geo_distance_from_home"] >= INTERNATIONAL_DISTANCE_KM_THRESHOLD
    ).astype(int)
    return df


# def add_card_present_proxy(df: pd.DataFrame) -> pd.DataFrame:
#     physical_merchant_keywords = [
#         "grocery",
#         "restaurant",
#         "fuel",
#         "gas",
#         "retail",
#         "pharmacy",
#         "supermarket",
#         "store",
#         "hotel",
#         "travel",
#     ]
#     merchant_type_text = df["merchant_type"].fillna("").astype(str).str.lower()
#     df["card_present_flag"] = merchant_type_text.apply(
#         lambda value: int(any(keyword in value for keyword in physical_merchant_keywords))
#     )
#     return df

def validate_features(df: pd.DataFrame) -> None:
    missing_features = [
        column for column in FINAL_FEATURE_COLUMNS
        if column not in df.columns
    ]
    if missing_features:
        raise ValueError(f"Missing final features: {missing_features}")
    
    if df[FINAL_FEATURE_COLUMNS].isna().any().any():
        missing_counts = df[FINAL_FEATURE_COLUMNS].isna().sum()
        raise ValueError(f"NaN values found in final features:\n{missing_counts}")
    
def build_velocity_rf_features() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    missing_input_columns = [
        column for column in REQUIRED_INPUT_COLUMNS
        if column not in df.columns
    ]

    if missing_input_columns:
        raise ValueError(f"Missing required input columns: {missing_input_columns}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["is_fraud"] = df["is_fraud"].astype(int)

    df = add_velocity_features(df)
    df = add_rich_velocity_features(df)
    df = add_geo_distance_from_home(df)
    df = add_is_international_proxy(df)
    df = add_amount_round_number(df)

    df["label"] = df["is_fraud"].astype(int)

    output_columns = [
        "timestamp",
        "transaction_id",
        "user_id",
        "amount",
        "city",
        "lat",
        "lon",
        "merchant_type",
        "merchant_id",
        "device_id",
        *FINAL_FEATURE_COLUMNS,
        "label",
    ]

    output_df = df[output_columns].copy()
    output_df[FINAL_FEATURE_COLUMNS] = (
        output_df[FINAL_FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    validate_features(output_df)
    output_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved Random Forest feature dataset to: {OUTPUT_PATH}")
    print(f"Rows: {len(output_df)}")
    print(f"Fraud rate: {output_df['label'].mean():.4f}")
    print("Feature summary:")
    print(output_df[FINAL_FEATURE_COLUMNS].describe())
    print("First rows:")
    print(output_df.head())


if __name__ == "__main__":
    build_velocity_rf_features()