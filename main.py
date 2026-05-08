import os
import json
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
import redis
from neo4j import GraphDatabase
from dotenv import load_dotenv
import time
from pathlib import Path
import sys
from agents.synthesis_agent import SynthesisAgent
from agents.anomaly_agent import AnomalyAgent
from agents.behaviour_agent import BehaviourAgent
from agents.risk_agent import RiskAgent
from agents.velocity_rules_agent import VelocityRulesAgent
from agents.otp_interlock import OTPInterlock
from agents.sms_agent import SMSAgent
from agents.email_agent import EmailAgent
from agents.otp_manager import OTPManager
from orchestrator.fraud_orchestrator import FraudOrchestrator

sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

app = FastAPI(title="Fraud Detection Orchestrator", version="1.0.0")

redis_client: Optional[redis.Redis] = None
fraud_orchestrator: Optional[FraudOrchestrator] = None


def get_redis() -> redis.Redis:
    global redis_client
    if redis_client is None:
        redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=0,
            decode_responses=True,
        )
    return redis_client


def get_orchestrator(redis: redis.Redis = Depends(get_redis)) -> FraudOrchestrator:
    global fraud_orchestrator
    if fraud_orchestrator is None:
        synthesis_agent = SynthesisAgent()
        anomaly_agent = AnomalyAgent()
        behaviour_agent = BehaviourAgent()
        risk_agent = RiskAgent()
        velocity_agent = VelocityRulesAgent()
        otp_manager = OTPManager()
        sms_agent = SMSAgent()
        email_agent = EmailAgent()
        otp_interlock = OTPInterlock(
            otp_manager=otp_manager,
            sms_agent=sms_agent,
            email_agent=email_agent,
        )
        fraud_orchestrator = FraudOrchestrator(
            synthesis_agent=synthesis_agent,
            anomaly_agent=anomaly_agent,
            behaviour_agent=behaviour_agent,
            risk_agent=risk_agent,
            velocity_agent=velocity_agent,
            otp_interlock=otp_interlock,
            redis_client=redis,
        )
    return fraud_orchestrator


class TransactionRequest(BaseModel):
    transaction_id: Optional[str] = None
    amount: float
    merchant_id: str
    user_id: str
    merchant_type: Optional[str] = None
    device_id: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    timestamp: Optional[str] = None


class OTPVerifyRequest(BaseModel):
    txn_id: str
    sms_code: str
    email_code: str


@app.on_event("startup")
async def startup():
    redis = get_redis()
    redis.ping()


@app.post("/transaction")
async def process_transaction(
    request: TransactionRequest,
    orchestrator: FraudOrchestrator = Depends(get_orchestrator),
):
    txn = request.dict(exclude_none=True)
    
    if txn.get("timestamp") is None:
        from datetime import datetime, timezone
        txn["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    redis = get_redis()
    redis.setex(f"txn:{txn.get('transaction_id', 'temp')}:context", 3600, json.dumps(txn))
    
    result = await orchestrator.process_transaction(txn)
    return result


@app.post("/otp/verify")
async def verify_otp(
    request: OTPVerifyRequest,
    orchestrator: FraudOrchestrator = Depends(get_orchestrator),
):
    result = await orchestrator.verify_otp(
        request.txn_id, request.sms_code, request.email_code
    )
    return result


@app.get("/transaction/{txn_id}/status")
async def get_transaction_status(
    txn_id: str,
    orchestrator: FraudOrchestrator = Depends(get_orchestrator),
):
    result = orchestrator.get_transaction_status(txn_id)
    return result


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": time.time()}
