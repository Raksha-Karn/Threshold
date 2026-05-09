import json
import asyncio
import logging
from typing import Dict, Any, Optional
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
        max_poll_records: int = 100,
        backpressure_warning_ms: float = 100.0,
    ):
        self.orchestrator = orchestrator
        self.input_topic = input_topic
        self.verdict_topic = verdict_topic
        self.dlq_topic = dlq_topic
        self.backpressure_warning_ms = backpressure_warning_ms
        self.logger = logging.getLogger(__name__)
        
        self.consumer = KafkaConsumer(
            input_topic,
            bootstrap_servers=bootstrap_servers.split(","),
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            max_poll_interval_ms=max_poll_interval_ms,
            max_poll_records=max_poll_records,
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
            "latency_ms_total": 0.0,
            "backpressure_warnings": 0,
        }

    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        try:
            start_time = time.perf_counter()
            result = await self.orchestrator.process_transaction(message)
            latency_ms = (time.perf_counter() - start_time) * 1000
            result.setdefault("pipeline_latency_ms", latency_ms)
            self.metrics["latency_ms_total"] += latency_ms
            if latency_ms > self.backpressure_warning_ms:
                self.metrics["backpressure_warnings"] += 1
                self.logger.warning(
                    "Orchestrator latency %.2fms exceeded %.2fms; scale the consumer group across topic partitions",
                    latency_ms,
                    self.backpressure_warning_ms,
                )
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
        metrics = self.metrics.copy()
        processed = metrics.get("processed", 0)
        metrics["avg_latency_ms"] = (
            metrics.get("latency_ms_total", 0.0) / processed
            if processed
            else 0.0
        )
        return metrics


class MockPaymentProcessor:
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        verdict_topic: str = "txn.verdict",
        group_id: str = "mock_payment_processor",
    ):
        self.logger = logging.getLogger(__name__)
        self.consumer = KafkaConsumer(
            verdict_topic,
            bootstrap_servers=bootstrap_servers.split(","),
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="earliest",
        )
        self.actions = {
            "processed": 0,
            "captured": 0,
            "blocked": 0,
            "held_for_otp": 0,
            "failed": 0,
        }

    def handle_verdict(self, verdict: Dict[str, Any]) -> str:
        decision = verdict.get("verdict")
        self.actions["processed"] += 1

        if decision == "APPROVED":
            self.actions["captured"] += 1
            return "CAPTURE_PAYMENT"
        if decision == "BLOCKED":
            self.actions["blocked"] += 1
            return "BLOCK_PAYMENT"
        if decision == "OTP_REQUIRED":
            self.actions["held_for_otp"] += 1
            return "HOLD_FOR_OTP"

        self.actions["failed"] += 1
        return "MANUAL_REVIEW"

    def run(self, max_messages: Optional[int] = None):
        message_count = 0
        try:
            for kafka_message in self.consumer:
                action = self.handle_verdict(kafka_message.value)
                self.logger.info("Payment action for %s: %s", kafka_message.value.get("txn_id"), action)
                message_count += 1
                if max_messages and message_count >= max_messages:
                    break
        finally:
            self.consumer.close()
