import asyncio
import time
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import redis
import mlflow
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from agents.synthesis_agent import SynthesisAgent
from agents.anomaly_agent import AnomalyAgent
from agents.behaviour_agent import BehaviourAgent
from agents.risk_agent import RiskAgent
from agents.velocity_rules_agent import VelocityRulesAgent
from agents.otp_interlock import OTPInterlock


class FraudOrchestrator:
    def __init__(
        self,
        synthesis_agent: SynthesisAgent,
        anomaly_agent: AnomalyAgent,
        behaviour_agent: BehaviourAgent,
        risk_agent: RiskAgent,
        velocity_agent: VelocityRulesAgent,
        otp_interlock: OTPInterlock,
        redis_client: redis.Redis,
    ):
        self.synthesis_agent = synthesis_agent
        self.anomaly_agent = anomaly_agent
        self.behaviour_agent = behaviour_agent
        self.risk_agent = risk_agent
        self.velocity_agent = velocity_agent
        self.otp_interlock = otp_interlock
        self.redis_client = redis_client
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def process_transaction(self, txn: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        txn_id = txn.get("transaction_id", f"txn_{int(time.time()*1000)}")
        
        try:
            txn["transaction_id"] = txn_id
            
            is_valid = self._validate_input(txn)
            if not is_valid:
                return {
                    "txn_id": txn_id,
                    "verdict": "DECLINED",
                    "reason": "invalid_input",
                    "latency_ms": (time.time() - start_time) * 1000,
                }
            
            velocity_result = self._check_velocity(txn)
            if velocity_result.get("blocked"):
                return {
                    "txn_id": txn_id,
                    "verdict": "BLOCKED",
                    "reason": "velocity_rule_triggered",
                    "latency_ms": (time.time() - start_time) * 1000,
                }
            
            fraud_score_result = await self.synthesis_agent.score_transaction(txn)

            if isinstance(fraud_score_result, dict):
                fraud_score = float(fraud_score_result.get("final_score", 0.0))
            else:
                fraud_score = float(getattr(fraud_score_result, "score", fraud_score_result))
                        
            agents_scores = await self._run_agents_concurrent(txn)
            
            all_scores = {
                "synthesis": fraud_score,
                "anomaly": agents_scores.get("anomaly", 0.0),
                "behaviour": agents_scores.get("behaviour", 0.0),
                "risk": agents_scores.get("risk", 0.0),
                "velocity": agents_scores.get("velocity", 0.0),
            }

            final_score = all_scores["synthesis"]
            
            final_score = all_scores["synthesis"]
            
            self._log_prediction(txn_id, txn, all_scores)
            
            if final_score < 0.3:
                verdict = "APPROVED"
            elif final_score < 0.7:
                verdict = "OTP_REQUIRED"
            else:
                verdict = "BLOCKED"
            
            result = {
                "txn_id": txn_id,
                "verdict": verdict,
                "score": final_score,
                "scores": all_scores,
                "latency_ms": (time.time() - start_time) * 1000,
            }
            
            if verdict == "OTP_REQUIRED":
                otp_result = await self._initiate_otp(txn)
                result["otp_status"] = otp_result
            
            self.redis_client.setex(f"txn:{txn_id}:verdict", 3600, str(verdict))
            
            return result
            
        except Exception as e:
            mlflow.log_param("error", str(e))
            return {
                "txn_id": txn_id,
                "verdict": "ERROR",
                "error": str(e),
                "latency_ms": (time.time() - start_time) * 1000,
            }

    def _validate_input(self, txn: Dict[str, Any]) -> bool:
        required = ["amount", "merchant_id", "user_id"]
        return all(field in txn for field in required)

    def _check_velocity(self, txn: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if hasattr(self.velocity_agent, "check_transaction"):
                result = self.velocity_agent.check_transaction(txn)

                if isinstance(result, dict):
                    return result

                return {"blocked": bool(result)}

            if hasattr(self.velocity_agent, "is_suspicious"):
                return {"blocked": bool(self.velocity_agent.is_suspicious(txn))}

            return {}

        except Exception:
            return {}

    async def _run_agents_concurrent(self, txn: Dict[str, Any]) -> Dict[str, float]:
        def extract_score(value: Any) -> float:
            if isinstance(value, Exception):
                return 0.0

            if isinstance(value, dict):
                return float(value.get("final_score", value.get("score", 0.0)))

            if hasattr(value, "score"):
                return float(value.score)

            try:
                return float(value)
            except Exception:
                return 0.0

        try:
            loop = asyncio.get_running_loop()

            anomaly_task = loop.run_in_executor(
                self.executor,
                self.anomaly_agent.score_transaction,
                txn,
            )

            behaviour_task = loop.run_in_executor(
                self.executor,
                self.behaviour_agent.score_transaction,
                txn,
            )

            risk_task = loop.run_in_executor(
                self.executor,
                self.risk_agent.score_transaction,
                txn,
            )

            velocity_task = loop.run_in_executor(
                self.executor,
                self.velocity_agent.score_transaction,
                txn,
            )

            scores = await asyncio.gather(
                anomaly_task,
                behaviour_task,
                risk_task,
                velocity_task,
                return_exceptions=True,
            )

            return {
                "anomaly": extract_score(scores[0]),
                "behaviour": extract_score(scores[1]),
                "risk": extract_score(scores[2]),
                "velocity": extract_score(scores[3]),
            }

        except Exception:
            return {
                "anomaly": 0.0,
                "behaviour": 0.0,
                "risk": 0.0,
                "velocity": 0.0,
            }
    async def _initiate_otp(self, txn: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = await self.otp_interlock.send_dual_otp(txn)
            return result
        except Exception as e:
            return {"error": str(e)}

    def _log_prediction(self, txn_id: str, txn: Dict[str, Any], scores: Dict[str, float]):
        try:
            mlflow.set_experiment("fraud_predictions")
            with mlflow.start_run(run_name=f"txn_{txn_id}"):
                mlflow.log_param("txn_id", txn_id)
                mlflow.log_param("user_id", txn.get("user_id"))
                mlflow.log_param("amount", txn.get("amount"))
                mlflow.log_param("merchant_id", txn.get("merchant_id"))
                for agent, score in scores.items():
                    mlflow.log_metric(f"score_{agent}", float(score))
        except Exception:
            pass

    async def verify_otp(self, txn_id: str, sms_code: str, email_code: str) -> Dict[str, Any]:
        try:
            txn = self._load_txn_context(txn_id)
            
            sms_result = await self.otp_interlock.verify_sms(txn_id, sms_code, txn)
            email_result = await self.otp_interlock.verify_email(txn_id, email_code, txn)
            
            if sms_result.get("success") and email_result.get("success"):
                verdict = "APPROVED"
            else:
                verdict = "FAILED"
            
            result = {
                "txn_id": txn_id,
                "verdict": verdict,
                "sms_status": sms_result.get("status"),
                "email_status": email_result.get("status"),
            }
            
            self.redis_client.setex(f"txn:{txn_id}:otp_verdict", 3600, verdict)
            return result
            
        except Exception as e:
            return {
                "txn_id": txn_id,
                "verdict": "ERROR",
                "error": str(e),
            }

    def _load_txn_context(self, txn_id: str) -> Dict[str, Any]:
        cached = self.redis_client.get(f"txn:{txn_id}:context")
        if cached:
            import json
            return json.loads(cached)
        return {"transaction_id": txn_id}

    def get_transaction_status(self, txn_id: str) -> Dict[str, Any]:
        verdict = self.redis_client.get(f"txn:{txn_id}:verdict")
        otp_status = self.otp_interlock.get_status(txn_id)
        
        return {
            "txn_id": txn_id,
            "verdict": verdict if verdict else None,
            "otp": otp_status,
        }
