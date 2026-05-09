import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.geo_agent import GeoAgent
from agents.graph_agent import GraphAgent
from agents.otp_interlock import OTPInterlock
from agents.otp_manager import OTPManager
from agents.velocity_rules_agent import VelocityRulesAgent
from orchestrator.fraud_orchestrator import FraudOrchestrator
from fakes import FakeRedis


class StaticSynthesisAgent:
    def __init__(self, score):
        self.score = score

    async def score_transaction(self, transaction):
        return {"final_score": self.score}


def build_orchestrator(score, redis_client=None, otp_interlock=None):
    redis_client = redis_client or FakeRedis()
    otp_interlock = otp_interlock or MagicMock()
    otp_interlock.send_dual_otp = AsyncMock(return_value={"sms_sent": True, "email_sent": True})
    otp_interlock.get_status = MagicMock(return_value={"status": "PENDING_DUAL"})
    if not hasattr(otp_interlock, "is_frozen"):
        otp_interlock.is_frozen = MagicMock(return_value=False)
    anomaly_agent = MagicMock()
    anomaly_agent.score_transaction = MagicMock(return_value=score)
    behaviour_agent = MagicMock()
    behaviour_agent.score_transaction = MagicMock(return_value=score)
    risk_agent = MagicMock()
    risk_agent.score_transaction = MagicMock(return_value=score)
    velocity_agent = MagicMock()
    velocity_agent.score_transaction = MagicMock(return_value=score)
    velocity_agent.check_transaction = MagicMock(return_value={"blocked": False})

    return FraudOrchestrator(
        synthesis_agent=StaticSynthesisAgent(score),
        anomaly_agent=anomaly_agent,
        behaviour_agent=behaviour_agent,
        risk_agent=risk_agent,
        velocity_agent=velocity_agent,
        otp_interlock=otp_interlock,
        redis_client=redis_client,
    )


@pytest.mark.asyncio
async def test_scenario_1_legit_2am_flight_ticket_not_blocked():
    orchestrator = build_orchestrator(score=0.42)
    transaction = {
        "transaction_id": "scenario_legit_flight",
        "user_id": "trusted_user",
        "amount": 1200.0,
        "merchant_id": "airline_001",
        "merchant_type": "travel",
        "txn_type": "ONLINE_PURCHASE",
        "timestamp": "2026-05-09T02:00:00+00:00",
        "phone": "+15550000001",
        "email": "trusted@example.com",
    }

    result = await orchestrator.process_transaction(transaction)

    assert result["verdict"] in {"APPROVED", "OTP_REQUIRED"}
    assert result["verdict"] != "BLOCKED"


@pytest.mark.asyncio
async def test_scenario_2_cold_start_large_transfer_requires_otp_minimum():
    orchestrator = build_orchestrator(score=0.65)
    transaction = {
        "transaction_id": "scenario_cold_start",
        "user_id": "brand_new_user",
        "amount": 5000.0,
        "merchant_id": "bank_transfer",
        "txn_type": "P2P_TRANSFER",
        "account_age_days": 0,
        "phone": "+15550000002",
        "email": "new@example.com",
    }

    result = await orchestrator.process_transaction(transaction)

    assert result["verdict"] in {"OTP_REQUIRED", "BLOCKED"}


@pytest.mark.asyncio
async def test_scenario_3_recent_sim_swap_fast_otp_blocks_and_escalates():
    redis_client = FakeRedis()
    otp_manager = OTPManager(redis_client=redis_client)
    otp_interlock = OTPInterlock(
        otp_manager=otp_manager,
        sms_agent=MagicMock(),
        email_agent=MagicMock(),
        neo4j_driver=MagicMock(),
    )
    orchestrator = build_orchestrator(
        score=0.55,
        redis_client=redis_client,
        otp_interlock=otp_interlock,
    )
    txn_id = "scenario_sim_swap"
    transaction = {
        "transaction_id": txn_id,
        "user_id": "sim_swap_user",
        "amount": 750.0,
        "merchant_id": "merchant_risky",
        "phone": "+15550000003",
        "email": "swap@example.com",
    }
    redis_client.setex(f"txn:{txn_id}:context", 3600, json.dumps(transaction))
    codes = otp_manager.generate_otp(txn_id)

    with patch.object(otp_interlock, "_is_recent_sim_swap", return_value=True):
        with patch.object(otp_interlock, "_notify_slack", new_callable=AsyncMock):
            with patch.object(otp_interlock, "_log_event"):
                with patch.object(otp_interlock, "_current_timestamp") as now:
                    first = datetime.now(timezone.utc)
                    now.side_effect = [
                        first.isoformat(),
                        (first + timedelta(seconds=10)).isoformat(),
                    ]
                    result = await orchestrator.verify_otp(txn_id, codes["sms"], codes["email"])

    assert result["verdict"] == "BLOCKED"
    assert redis_client.get(f"otp:{txn_id}:frozen") == "1"


def test_scenario_4_fraud_ring_shared_device_detected_by_graph_agent():
    transactions = [
        {"user_id": "ring_user_1", "device_id": "device_shared", "amount": 900.0},
        {"user_id": "ring_user_2", "device_id": "device_shared", "amount": 950.0},
        {"user_id": "ring_user_3", "device_id": "device_shared", "amount": 990.0},
    ]

    result = GraphAgent.detect_shared_device_ring(transactions)

    assert result["flagged"] is True
    assert result["score"] >= 0.9
    assert result["risky_devices"]["device_shared"] == [
        "ring_user_1",
        "ring_user_2",
        "ring_user_3",
    ]


def test_scenario_5_geo_impossible_kathmandu_to_london_flags():
    agent = GeoAgent()
    kathmandu_txn = {
        "user_id": "geo_user",
        "lat": 27.7172,
        "lon": 85.3240,
        "timestamp": "2026-05-09T10:00:00+00:00",
    }
    london_txn = {
        "user_id": "geo_user",
        "lat": 51.5074,
        "lon": -0.1278,
        "timestamp": "2026-05-09T10:30:00+00:00",
    }

    agent.score(kathmandu_txn)
    score = agent.score(london_txn)

    assert score >= 0.5


def test_scenario_6_velocity_15_transactions_in_10_minutes_flags():
    redis_client = FakeRedis()
    agent = VelocityRulesAgent()
    agent.redis_client = redis_client
    base_timestamp = 1770000000.0
    result = None

    for index in range(15):
        result = agent.check_transaction(
            {
                "user_id": "velocity_user",
                "amount": 99.0,
                "merchant_id": f"merchant_{index}",
                "device_id": "velocity_device",
                "timestamp_epoch": base_timestamp + index * 40,
            }
        )

    assert result["blocked"] is True
    assert result["reason"] == "velocity_threshold_exceeded"
    assert result["count"] == 15
