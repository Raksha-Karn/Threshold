from pathlib import Path
import pandas as pd
import numpy as np

RAW_DATA_PATH = Path("data/raw/transactions_with_anomalies.csv")
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
    "geo_distance_from_home",
    "is_international",
    "merchant_risk_score",
    "amount_round_number",
    "card_present_flag",
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
    df["txns_last_1h"] = df["txn_count_in_last_1h"].fillna(0).astype(float)
    df["txns_last_24h"] = df["txn_count_in_last_24h"].fillna(0).astype(float)
    return df

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


def add_card_present_proxy(df: pd.DataFrame) -> pd.DataFrame:
    physical_merchant_keywords = [
        "grocery",
        "restaurant",
        "fuel",
        "gas",
        "retail",
        "pharmacy",
        "supermarket",
        "store",
        "hotel",
        "travel",
    ]
    merchant_type_text = df["merchant_type"].fillna("").astype(str).str.lower()
    df["card_present_flag"] = merchant_type_text.apply(
        lambda value: int(any(keyword in value for keyword in physical_merchant_keywords))
    )
    return df