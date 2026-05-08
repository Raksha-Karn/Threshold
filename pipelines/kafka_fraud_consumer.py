import json
import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
import time


class KafkaFraudConsumer:
    def __init__(
        self,
        orchestrator: Any,
        bootstrap_servers: str = "localhost:9092",
        input_topic: str = "txn.raw",
        verdict_topic: str = "txn.verdict",
        dlq_topic: str = "txn.dlq",
        group_id: str = "fraud_detection_group",
        max_poll_interval_ms: int = 600000,
    ):
        self.orchestrator = orchestrator
        self.input_topic = input_topic
        self.verdict_topic = verdict_topic
        self.dlq_topic = dlq_topic
        self.logger = logging.getLogger(__name__)
        
        self.consumer = KafkaConsumer(
            input_topic,
            bootstrap_servers=bootstrap_servers.split(","),
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            max_poll_interval_ms=max_poll_interval_ms,
            auto_offset_reset="earliest",
        )
        
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers.split(","),
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            retries=3,
        )
        
        self.metrics = {
            "processed": 0,
            "approved": 0,
            "blocked": 0,
            "otp_required": 0,
            "errors": 0,
            "dlq_sent": 0,
        }

    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = await self.orchestrator.process_transaction(message)
            return result
        except Exception as e:
            self.logger.error(f"Error processing transaction: {str(e)}")
            raise

    def publish_verdict(self, verdict: Dict[str, Any]):
        try:
            self.producer.send(self.verdict_topic, value=verdict).get(timeout=10)
            self.metrics["processed"] += 1
            
            if verdict.get("verdict") == "APPROVED":
                self.metrics["approved"] += 1
            elif verdict.get("verdict") == "BLOCKED":
                self.metrics["blocked"] += 1
            elif verdict.get("verdict") == "OTP_REQUIRED":
                self.metrics["otp_required"] += 1
                
        except KafkaError as e:
            self.logger.error(f"Failed to publish verdict: {str(e)}")

    def publish_dlq(self, message: Dict[str, Any], error: str):
        try:
            dlq_message = {
                "original_message": message,
                "error": error,
                "timestamp": time.time(),
            }
            self.producer.send(self.dlq_topic, value=dlq_message).get(timeout=10)
            self.metrics["dlq_sent"] += 1
        except KafkaError as e:
            self.logger.error(f"Failed to publish to DLQ: {str(e)}")

    def run(self, max_messages: Optional[int] = None):
        message_count = 0
        
        try:
            for kafka_message in self.consumer:
                try:
                    txn = kafka_message.value
                    
                    result = asyncio.run(self.process_message(txn))
                    
                    self.publish_verdict(result)
                    
                    message_count += 1
                    if max_messages and message_count >= max_messages:
                        break
                        
                except Exception as e:
                    self.logger.error(f"Error processing message: {str(e)}")
                    self.publish_dlq(kafka_message.value, str(e))
                    self.metrics["errors"] += 1
                    
        except KeyboardInterrupt:
            self.logger.info("Consumer interrupted")
        finally:
            self.consumer.close()
            self.producer.close()

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.copy()
