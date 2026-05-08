import json
from kafka import KafkaProducer
from kafka.errors import KafkaError
import time
from typing import Dict, Any, List


class KafkaFraudProducer:
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "txn.raw",
    ):
        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers.split(","),
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            retries=3,
        )

    def send_transaction(self, txn: Dict[str, Any]) -> bool:
        try:
            self.producer.send(self.topic, value=txn).get(timeout=10)
            return True
        except Exception as e:
            print(f"Failed to send transaction: {str(e)}")
            return False

    def send_batch(self, txns: List[Dict[str, Any]]) -> int:
        success_count = 0
        for txn in txns:
            if self.send_transaction(txn):
                success_count += 1
        self.producer.flush()
        return success_count

    def close(self):
        self.producer.close()
