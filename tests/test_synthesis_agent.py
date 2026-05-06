import asyncio
import time
import pytest
from agents.synthesis_agent import SynthesisAgent

@pytest.fixture
async def agent():
    return SynthesisAgent()

@pytest.mark.asyncio
async def test_false_positive_rate(agent):
    clean_txns = [
        {
            "user_id": "user1",
            "txn_type": "POS_RETAIL",
            "amount": 50.0,
            "lat": 40.7128,
            "lon": -74.0060,
            "timestamp": "2023-01-01T12:00:00Z",
            "user_tenure_years": 3,
            "clean_history": True,
            "merchant_id": "TRUSTED_SHOP_1",
        }
    ] * 100  

    scores = []
    for txn in clean_txns:
        result = await agent.score_transaction(txn)
        scores.append(result.score)

    fpr = sum(1 for s in scores if s >= 0.3) / len(scores)  # Using 0.3 as threshold
    assert fpr < 0.02, f"False positive rate {fpr} exceeds 2%"

@pytest.mark.asyncio
async def test_latency(agent):
    txn = {
        "user_id": "user1",
        "txn_type": "POS_RETAIL",
        "amount": 50.0,
        "lat": 40.7128,
        "lon": -74.0060,
        "timestamp": "2023-01-01T12:00:00Z",
    }

    start = time.perf_counter()
    result = await agent.score_transaction(txn)
    end = time.perf_counter()

    latency_ms = (end - start) * 1000
    assert latency_ms < 300, f"Latency {latency_ms}ms exceeds 300ms"

@pytest.mark.asyncio
async def test_verdict_categories(agent):
    txn = {
        "user_id": "user1",
        "txn_type": "POS_RETAIL",
        "amount": 50.0,
        "lat": 40.7128,
        "lon": -74.0060,
        "timestamp": "2023-01-01T12:00:00Z",
    }

    result = await agent.score_transaction(txn)
    assert result.verdict in ["APPROVE", "REVIEW", "OTP_REQUIRED", "BLOCK"]
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0