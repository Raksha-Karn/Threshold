import os
import json
from typing import Optional, Any
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import Response, HTMLResponse
from pydantic import BaseModel
import redis
from neo4j import GraphDatabase
from dotenv import load_dotenv
import time
from pathlib import Path
import sys
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from monitoring.metrics import (
    REQUEST_LATENCY,
    FRAUD_SCORE_HISTOGRAM,
    VERDICT_COUNTER,
    OTP_SUCCESS_COUNTER,
    OTP_FAILURE_COUNTER,
)
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
neo4j_driver: Optional[Any] = None
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


def get_neo4j_driver() -> Any:
    global neo4j_driver
    if neo4j_driver is None:
        neo4j_driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.getenv("NEO4J_USER", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "password"),
            ),
        )
    return neo4j_driver


def get_orchestrator(
    redis: redis.Redis = Depends(get_redis),
    neo4j: Any = Depends(get_neo4j_driver),
) -> FraudOrchestrator:
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
            neo4j_driver=neo4j,
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


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - start_time) * 1000
    REQUEST_LATENCY.observe(elapsed_ms)
    return response


@app.on_event("startup")
async def startup():
    get_redis().ping()
    get_neo4j_driver()


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


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/dashboard")
def dashboard() -> HTMLResponse:
    html = """
    <html>
      <head>
        <title>Fraud Orchestrator Dashboard</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 24px; }
          .metric { margin-bottom: 16px; }
          pre { background: #f4f4f4; padding: 16px; border-radius: 8px; }
        </style>
      </head>
      <body>
        <h1>Fraud Orchestrator Dashboard</h1>
        <div class="metric"><strong>Metrics Endpoint:</strong> <a href="/metrics">/metrics</a></div>
        <div class="metric"><strong>Latest Metrics</strong></div>
        <pre id="metrics">Loading metrics...</pre>
        <script>
          async function loadMetrics() {
            const response = await fetch('/metrics');
            const text = await response.text();
            document.getElementById('metrics').textContent = text;
          }
          loadMetrics();
          setInterval(loadMetrics, 5000);
        </script>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": time.time()}
