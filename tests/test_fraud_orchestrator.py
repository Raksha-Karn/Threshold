import pytest
import sys
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.fraud_orchestrator import FraudOrchestrator
from agents.synthesis_agent import SynthesisAgent
from agents.anomaly_agent import AnomalyAgent
from agents.behaviour_agent import BehaviourAgent
from agents.risk_agent import RiskAgent
from agents.velocity_rules_agent import VelocityRulesAgent
from agents.otp_interlock import OTPInterlock


@pytest.fixture
def mock_redis():
    redis = MagicMock()
    redis.setex = MagicMock()
    redis.get = MagicMock(return_value=None)
    return redis


@pytest.fixture
def mock_agents():
    synthesis = AsyncMock(spec=SynthesisAgent)
    synthesis.score_transaction = AsyncMock(return_value={"final_score": 0.5})
    
    anomaly = MagicMock(spec=AnomalyAgent)
    anomaly.score_transaction = MagicMock(return_value=0.4)
    
    behaviour = MagicMock(spec=BehaviourAgent)
    behaviour.score_transaction = MagicMock(return_value=0.3)
    
    risk = MagicMock(spec=RiskAgent)
    risk.score_transaction = MagicMock(return_value=0.2)
    
    velocity = MagicMock(spec=VelocityRulesAgent)
    velocity.score_transaction = MagicMock(return_value=0.1)
    velocity.check_transaction = MagicMock(return_value={})
    
    return {
        "synthesis": synthesis,
        "anomaly": anomaly,
        "behaviour": behaviour,
        "risk": risk,
        "velocity": velocity,
    }


@pytest.fixture
def mock_otp_interlock():
    otp = AsyncMock(spec=OTPInterlock)
    otp.send_dual_otp = AsyncMock(return_value={"sms_sent": True, "email_sent": True})
    otp.verify_sms = AsyncMock(return_value={"success": True, "status": "SMS_CONFIRMED"})
    otp.verify_email = AsyncMock(return_value={"success": True, "status": "EMAIL_CONFIRMED"})
    otp.get_status = MagicMock(return_value={"status": "PENDING_DUAL"})
    return otp


@pytest.fixture
def orchestrator(mock_redis, mock_agents, mock_otp_interlock):
    return FraudOrchestrator(
        synthesis_agent=mock_agents["synthesis"],
        anomaly_agent=mock_agents["anomaly"],
        behaviour_agent=mock_agents["behaviour"],
        risk_agent=mock_agents["risk"],
        velocity_agent=mock_agents["velocity"],
        otp_interlock=mock_otp_interlock,
        redis_client=mock_redis,
    )


class TestFraudOrchestrator:
    def test_validate_input_valid(self, orchestrator):
        txn = {"amount": 100.0, "merchant_id": "M1", "user_id": "U1"}
        assert orchestrator._validate_input(txn) is True

    def test_validate_input_missing_field(self, orchestrator):
        txn = {"amount": 100.0, "merchant_id": "M1"}
        assert orchestrator._validate_input(txn) is False

    def test_check_velocity_no_block(self, orchestrator):
        txn = {"user_id": "U1", "amount": 50.0}
        result = orchestrator._check_velocity(txn)
        assert result.get("blocked") is None or result.get("blocked") is False

    @pytest.mark.asyncio
    async def test_process_transaction_low_risk(self, orchestrator):
        orchestrator.synthesis_agent.score_transaction = AsyncMock(
            return_value={"final_score": 0.2}
        )
        
        txn = {
            "amount": 100.0,
            "merchant_id": "M1",
            "user_id": "U1",
            "device_id": "D1",
            "phone": "+1234567890",
            "email": "user@example.com",
        }
        
        result = await orchestrator.process_transaction(txn)
        
        assert result["verdict"] == "APPROVED"
        assert result["score"] == 0.2
        assert "latency_ms" in result

    @pytest.mark.asyncio
    async def test_process_transaction_medium_risk(self, orchestrator):
        orchestrator.synthesis_agent.score_transaction = AsyncMock(
            return_value={"final_score": 0.5}
        )
        
        txn = {
            "amount": 500.0,
            "merchant_id": "M2",
            "user_id": "U2",
            "device_id": "D2",
            "phone": "+9876543210",
            "email": "user2@example.com",
        }
        
        result = await orchestrator.process_transaction(txn)
        
        assert result["verdict"] == "OTP_REQUIRED"
        assert result["score"] == 0.5
        assert "otp_status" in result

    @pytest.mark.asyncio
    async def test_process_transaction_high_risk(self, orchestrator):
        orchestrator.synthesis_agent.score_transaction = AsyncMock(
            return_value={"final_score": 0.8}
        )
        
        txn = {
            "amount": 5000.0,
            "merchant_id": "M3",
            "user_id": "U3",
        }
        
        result = await orchestrator.process_transaction(txn)
        
        assert result["verdict"] == "BLOCKED"
        assert result["score"] == 0.8

    @pytest.mark.asyncio
    async def test_process_transaction_invalid_input(self, orchestrator):
        txn = {"amount": 100.0}
        
        result = await orchestrator.process_transaction(txn)
        
        assert result["verdict"] == "DECLINED"
        assert result["reason"] == "invalid_input"

    @pytest.mark.asyncio
    async def test_process_transaction_velocity_blocked(self, orchestrator):
        orchestrator.velocity_agent.check_transaction = MagicMock(
            return_value={"blocked": True}
        )
        
        txn = {
            "amount": 100.0,
            "merchant_id": "M1",
            "user_id": "U1",
        }
        
        result = await orchestrator.process_transaction(txn)
        
        assert result["verdict"] == "BLOCKED"
        assert result["reason"] == "velocity_rule_triggered"

    @pytest.mark.asyncio
    async def test_run_agents_concurrent(self, orchestrator):
        txn = {"user_id": "U1", "amount": 100.0}
        
        scores = await orchestrator._run_agents_concurrent(txn)
        
        assert "anomaly" in scores
        assert "behaviour" in scores
        assert "risk" in scores
        assert "velocity" in scores
        assert all(isinstance(v, float) for v in scores.values())

    @pytest.mark.asyncio
    async def test_verify_otp_success(self, orchestrator, mock_redis):
        mock_redis.get.return_value = json.dumps({
            "transaction_id": "txn_123",
            "user_id": "U1",
            "phone": "+1234567890",
            "email": "user@example.com",
        })
        
        result = await orchestrator.verify_otp("txn_123", "123456", "654321")
        
        assert result["verdict"] == "APPROVED"
        assert result["sms_status"] == "SMS_CONFIRMED"
        assert result["email_status"] == "EMAIL_CONFIRMED"

    @pytest.mark.asyncio
    async def test_verify_otp_failure(self, orchestrator, mock_redis):
        orchestrator.otp_interlock.verify_sms = AsyncMock(
            return_value={"success": False, "status": "FAILED"}
        )
        orchestrator.otp_interlock.verify_email = AsyncMock(
            return_value={"success": False, "status": "FAILED"}
        )
        mock_redis.get.return_value = json.dumps({
            "transaction_id": "txn_fail",
            "user_id": "U1",
        })
        
        result = await orchestrator.verify_otp("txn_fail", "000000", "000000")
        
        assert result["verdict"] == "FAILED"

    def test_get_transaction_status(self, orchestrator, mock_redis):
        mock_redis.get.return_value = "APPROVED"
        
        result = orchestrator.get_transaction_status("txn_123")
        
        assert result["txn_id"] == "txn_123"
        assert result["verdict"] == "APPROVED"
        assert "otp" in result

    def test_get_transaction_status_not_found(self, orchestrator, mock_redis):
        mock_redis.get.return_value = None
        
        result = orchestrator.get_transaction_status("txn_notfound")
        
        assert result["txn_id"] == "txn_notfound"
        assert result["verdict"] is None

    @pytest.mark.asyncio
    async def test_process_transaction_latency(self, orchestrator):
        txn = {
            "amount": 100.0,
            "merchant_id": "M1",
            "user_id": "U1",
        }
        
        result = await orchestrator.process_transaction(txn)
        
        assert result["latency_ms"] < 5000
        assert result["latency_ms"] > 0

    @pytest.mark.asyncio
    async def test_process_transaction_exception_handling(self, orchestrator):
        orchestrator.synthesis_agent.score_transaction = AsyncMock(
            side_effect=Exception("Test error")
        )
        
        txn = {
            "amount": 100.0,
            "merchant_id": "M1",
            "user_id": "U1",
        }
        
        result = await orchestrator.process_transaction(txn)
        
        assert result["verdict"] == "ERROR"
        assert "error" in result
