import json 
import time
import uuid
import random
from datetime import datetime, timezone
import signal
from confluent_kafka import Producer

TOPIC = "txn.raw"
RUNNING = True
def handle_shutdown(signum, frame):
    global RUNNING
    RUNNING = False

signal.signal(signalnum=signal.SIGINT, handler=handle_shutdown)
signal.signal(signalnum=signal.SIGTERM, handler=handle_shutdown)

producer_config = {
    "bootstrap.servers": "localhost:9092",
    "client.id": "fraud-transaction-producer",
    "acks": "all",
    "batch.num.messages": 1000,
    "linger.ms": 10,
    "enable.idempotence": True,
    "compression.type": "lz4"
}

producer = Producer(producer_config)

def delivery_report(err, msg):
    if err is not None:
        print(f"FAILED: {err}")
        return
    print(f"TOPIC = {msg.topic()}")
    print(f"PARTITION = {msg.partition()}")
    print(f"OFFSET = {msg.offset()}")

def generate_transaction():
    user_id = f"user_{random.randint(1, 500)}"
    is_fraud = random.random() < 0.12
    txn = {
        "transaction_id": str(uuid.uuid4()),
        "user_id": user_id,
        "account_id": f"acct_{random.randint(1, 200)}",
        "merchant_id": f"merchant_{random.randint(1, 80)}",
        "amount": round(random.uniform(5, 700), 2),
        "currency": "USD",
        "country": random.choice(["NP", "IN", "US", "GB", "AE"]),
        "device_id": f"device_{random.randint(1, 400)}",
        "payment_method": random.choice(["card", "wallet", "bank_transfer"]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": "legit"
    }

    if is_fraud:
        txn["amount"] = round(random.uniform(900,8000), 2)
        txn["merchant_id"] = "merchant_unknown"
        txn["country"] = random.choice(["RU", "CN", "Unknown"])
        txn["device_id"] = f"new_device_{uuid.uuid4()}"
        txn["label"] = "fraud"

    return txn

try:
    while RUNNING:
        txn = generate_transaction()
        producer.produce(
            topic=TOPIC,
            key=txn["user_id"],
            value=json.dumps(txn).encode("utf-8"),
            callback=delivery_report
        )
        producer.poll(0)
        print("Queued: ", txn)
        time.sleep(0.5)
except BufferError:
    print("Producer queue is full. Kafka may be slow or unavailable!")
finally:
    print("Flushing remaining messages...")
    producer.flush()
    print("Producer stopped.")



