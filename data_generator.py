import random
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
import csv

NUM_TRANSACTIONS = 100000
NUM_USERS = 10000
NUM_DEVICES = 7500
FRAUD_RATE = 0.2
OUTPUT_PATH_CSV = Path("data/transactions.csv")

CITIES = [
    ("Mumbai", 19.0760, 72.8777),
    ("Delhi", 28.6139, 77.2090),
    ("Bengaluru", 12.9716, 77.5946),
    ("Hyderabad", 17.3850, 78.4867),
    ("Chennai", 13.0827, 80.2707),
    ("Pune", 18.5204, 73.8567),
]

FRAUD_MERCHANTS = ["crypto", "atm", "electronics", "jewelry", "gaming"]
LEGIT_MERCHANTS = ["grocery", "fuel", "restaurant", "fashion", "pharmacy", "travel"]

NUM_FRAUD_MERCHANTS = 300
NUM_LEGIT_MERCHANTS = 700

FRAUD_MERCHANT_POOL = [
    {
        "merchant_id": f"fraud_merchant_{i}",
        "merchant_type": random.choice(FRAUD_MERCHANTS)
    }
    for i in range(1, NUM_FRAUD_MERCHANTS + 1)
]

LEGIT_MERCHANT_POOL = [
    {
        "merchant_id": f"legit_merchant_{i}",
        "merchant_type": random.choice(LEGIT_MERCHANTS)
    }
    for i in range(1, NUM_LEGIT_MERCHANTS + 1)
]

def random_timestamp(days_back=90, fraud=False):
    now = datetime.now(timezone.utc)
    if fraud:
        hour = random.choice([0,1,2,3,4,23])
    else:
        hour = random.randint(5,22)
    fake_time = now - timedelta(
        days=random.randint(0, days_back),
        hours=hour,
        minutes=random.randint(0,59),
        seconds=random.randint(0,59)
    )

    return fake_time.isoformat()

def random_location(fraud=False):
    city, lat, lon = random.choice(CITIES)

    if fraud:
        spread = 0.20
    else:
        spread = 0.04

    return {
        "city": city,
        "lat": round(lat + random.uniform(-spread, spread), 6),
        "lon": round(lon + random.uniform(-spread, spread), 6),
    }

def random_amount(merchant_type, fraud=False):
    if fraud:
        if merchant_type in ["crypto", "jewelry", "electronics"]:
            return round(random.uniform(15000, 1500000), 2)
        if merchant_type == "atm":
            return round(random.uniform(8000, 500000), 2)
        return round(random.uniform(5000, 800000), 2)
    if merchant_type in ["grocery", "pharmacy"]:
        return round(random.uniform(100, 5000), 2)
    if merchant_type in ["restaurant", "fuel", "fashion"]:
        return round(random.uniform(200, 8000), 2)
    if merchant_type in ["electronics", "travel"]:
        return round(random.uniform(3000, 70000), 2)
    return round(random.uniform(200, 12000), 2)

def generate_transactions(is_fraud):
    user_id = f"user_{random.randint(1, NUM_USERS)}"

    if is_fraud:
        device_id = f"device_{random.randint(1, 1200)}"
        merchant = random.choice(FRAUD_MERCHANT_POOL)
    else:
        device_id = f"device_{random.randint(1, NUM_DEVICES)}"
        merchant = random.choice(LEGIT_MERCHANT_POOL)

    location = random_location(fraud=is_fraud)
    merchant_id = merchant["merchant_id"]
    merchant_type = merchant["merchant_type"]

    return {
        "transaction_id": str(uuid.uuid4()),
        "user_id": user_id,
        "amount": random_amount(merchant_type=merchant_type, fraud=is_fraud),
        "city": location["city"], 
        "lat": location["lat"], 
        "lon": location["lon"],
        "merchant_type": merchant_type,
        "merchant_id": merchant_id,
        "timestamp": random_timestamp(fraud=is_fraud),
        "device_id": device_id,
        "is_fraud": int(is_fraud)
    }

def main():
    OUTPUT_PATH_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "transaction_id", 
        "user_id", 
        "amount", 
        "city", 
        "lat", 
        "lon", 
        "merchant_type", 
        "merchant_id",
        "timestamp", 
        "device_id", 
        "is_fraud"
    ]

    fraud_count = int(NUM_TRANSACTIONS * FRAUD_RATE)
    legit_count = NUM_TRANSACTIONS - fraud_count

    transactions = []
    for _ in range(legit_count):
        transactions.append(generate_transactions(is_fraud=False))
    for _ in range(fraud_count):
        transactions.append(generate_transactions(is_fraud=True))
    random.shuffle(transactions)

    with OUTPUT_PATH_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(transactions)

    print(f"Generated {NUM_TRANSACTIONS} transactions")
    print(f"Legit count: {legit_count}\t\tFraud count: {fraud_count}")
    print(f"Saved to {OUTPUT_PATH_CSV}")


if __name__ == "__main__":
    main()
