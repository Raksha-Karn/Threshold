import pandas as pd

df = pd.read_csv('data/transactions.csv', encoding='utf-8')

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["user_id", "timestamp"])

df["hour_of_day"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek

amount_mean = df["amount"].mean()
amount_std = df["amount"].std()
df["amount_zscore"] = (df["amount"] - amount_mean) / amount_std

df["user_mean_amount"] = df.groupby("user_id")["amount"].transform("mean")
df["amount_vs_user_mean"] = df["amount"] / df["user_mean_amount"]

df = df.set_index("timestamp")
df["txn_count_in_last_1h"] = df.groupby("user_id")["amount"].rolling("1h").count().reset_index(level=0, drop=True)
df["txn_count_in_last_24h"] = df.groupby("user_id")["amount"].rolling("24h").count().reset_index(level=0, drop=True)

df = df.reset_index()

seen_merchants = {}
is_new = []

for _, row in df.iterrows():
    user = row["user_id"]
    merchant = row["merchant_id"]
    if user not in seen_merchants:
       seen_merchants[user] = set()
    if merchant in seen_merchants[user]:
        is_new.append(0)
    else:
        is_new.append(1)
        seen_merchants[user].add(merchant)

df["is_new_merchant"] = is_new
df.to_csv("data/transaction_features.csv", index=False)
print(df.head())
print("Saved featured file")