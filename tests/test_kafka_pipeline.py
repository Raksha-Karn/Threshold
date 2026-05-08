import pytest
import sys
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.kafka_fraud_consumer import KafkaFraudConsumer
from pipelines.kafka_fraud_producer import KafkaFraudProducer


@pytest.fixture
def mock_orchestrator():
    orchestrator = AsyncMock()
    orchestrator.process_transaction = AsyncMock(
        return_value={"verdict": "APPROVED", "score": 0.3, "txn_id": "test_txn"}
    )
    return orchestrator


@pytest.fixture
def kafka_consumer(mock_orchestrator):
    with patch("pipelines.kafka_fraud_consumer.KafkaConsumer"):
        with patch("pipelines.kafka_fraud_consumer.KafkaProducer"):
            consumer = KafkaFraudConsumer(
                orchestrator=mock_orchestrator,
                input_topic="test.raw",
                verdict_topic="test.verdict",
                dlq_topic="test.dlq",
            )
            return consumer


@pytest.fixture
def kafka_producer():
    with patch("pipelines.kafka_fraud_producer.KafkaProducer"):
        producer = KafkaFraudProducer(topic="test.raw")
        return producer


class TestKafkaFraudConsumer:
    @pytest.mark.asyncio
    async def test_process_message_success(self, kafka_consumer):
        txn = {
            "transaction_id": "txn_001",
            "amount": 100.0,
            "merchant_id": "M1",
            "user_id": "U1",
        }
        
        result = await kafka_consumer.process_message(txn)
        
        assert result["verdict"] == "APPROVED"
        assert result["score"] == 0.3

    @pytest.mark.asyncio
    async def test_process_message_error(self, kafka_consumer):
        kafka_consumer.orchestrator.process_transaction = AsyncMock(
            side_effect=Exception("Orchestrator error")
        )
        
        txn = {"transaction_id": "txn_002"}
        
        with pytest.raises(Exception):
            await kafka_consumer.process_message(txn)

    def test_publish_verdict_approved(self, kafka_consumer):
        kafka_consumer.producer = MagicMock()
        kafka_consumer.producer.send = MagicMock(
            return_value=MagicMock(get=MagicMock(return_value=None))
        )
        
        verdict = {
            "txn_id": "txn_003",
            "verdict": "APPROVED",
            "score": 0.2,
        }
        
        kafka_consumer.publish_verdict(verdict)
        
        assert kafka_consumer.metrics["processed"] == 1
        assert kafka_consumer.metrics["approved"] == 1

    def test_publish_verdict_blocked(self, kafka_consumer):
        kafka_consumer.producer = MagicMock()
        kafka_consumer.producer.send = MagicMock(
            return_value=MagicMock(get=MagicMock(return_value=None))
        )
        
        verdict = {
            "txn_id": "txn_004",
            "verdict": "BLOCKED",
            "score": 0.8,
        }
        
        kafka_consumer.publish_verdict(verdict)
        
        assert kafka_consumer.metrics["processed"] == 1
        assert kafka_consumer.metrics["blocked"] == 1

    def test_publish_verdict_otp_required(self, kafka_consumer):
        kafka_consumer.producer = MagicMock()
        kafka_consumer.producer.send = MagicMock(
            return_value=MagicMock(get=MagicMock(return_value=None))
        )
        
        verdict = {
            "txn_id": "txn_005",
            "verdict": "OTP_REQUIRED",
            "score": 0.5,
        }
        
        kafka_consumer.publish_verdict(verdict)
        
        assert kafka_consumer.metrics["processed"] == 1
        assert kafka_consumer.metrics["otp_required"] == 1

    def test_publish_dlq(self, kafka_consumer):
        kafka_consumer.producer = MagicMock()
        kafka_consumer.producer.send = MagicMock(
            return_value=MagicMock(get=MagicMock(return_value=None))
        )
        
        message = {"txn_id": "txn_006"}
        error = "Test error message"
        
        kafka_consumer.publish_dlq(message, error)
        
        assert kafka_consumer.metrics["dlq_sent"] == 1
        kafka_consumer.producer.send.assert_called_once()
        call_args = kafka_consumer.producer.send.call_args
        assert call_args[0][0] == "test.dlq"

    def test_get_metrics(self, kafka_consumer):
        kafka_consumer.metrics = {
            "processed": 10,
            "approved": 7,
            "blocked": 2,
            "otp_required": 1,
            "errors": 0,
            "dlq_sent": 0,
        }
        
        metrics = kafka_consumer.get_metrics()
        
        assert metrics["processed"] == 10
        assert metrics["approved"] == 7
        assert metrics["blocked"] == 2
        assert metrics["otp_required"] == 1


class TestKafkaFraudProducer:
    def test_send_transaction_success(self, kafka_producer):
        kafka_producer.producer = MagicMock()
        kafka_producer.producer.send = MagicMock(
            return_value=MagicMock(get=MagicMock(return_value=None))
        )
        
        txn = {
            "transaction_id": "txn_prod_001",
            "amount": 100.0,
            "merchant_id": "M1",
            "user_id": "U1",
        }
        
        result = kafka_producer.send_transaction(txn)
        
        assert result is True
        kafka_producer.producer.send.assert_called_once()

    def test_send_transaction_failure(self, kafka_producer):
        kafka_producer.producer = MagicMock()
        kafka_producer.producer.send = MagicMock(
            return_value=MagicMock(get=MagicMock(side_effect=Exception("Kafka error")))
        )
        
        txn = {"transaction_id": "txn_prod_002"}
        
        result = kafka_producer.send_transaction(txn)
        
        assert result is False

    def test_send_batch(self, kafka_producer):
        kafka_producer.producer = MagicMock()
        kafka_producer.producer.send = MagicMock(
            return_value=MagicMock(get=MagicMock(return_value=None))
        )
        
        txns = [
            {"transaction_id": f"txn_{i}", "amount": 100.0}
            for i in range(5)
        ]
        
        success_count = kafka_producer.send_batch(txns)
        
        assert success_count == 5
        assert kafka_producer.producer.send.call_count == 5
        kafka_producer.producer.flush.assert_called_once()

    def test_close(self, kafka_producer):
        kafka_producer.producer = MagicMock()
        
        kafka_producer.close()
        
        kafka_producer.producer.close.assert_called_once()
