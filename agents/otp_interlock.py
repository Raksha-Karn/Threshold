import os
import json
import asyncio
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
import mlflow
from neo4j import GraphDatabase

from agents.otp_manager import OTPManager
from agents.sms_agent import SMSAgent
from agents.email_agent import EmailAgent


class OTPInterlock:
    def __init__(
        self,
        sms_agent: Optional[SMSAgent] = None,
        email_agent: Optional[EmailAgent] = None,
        otp_manager: Optional[OTPManager] = None,
        neo4j_driver: Optional[Any] = None,
        neo4j_uri: str | None = None,
        neo4j_user: str | None = None,
        neo4j_password: str | None = None,
    ):
        self.otp_manager = otp_manager or OTPManager()
        self.sms_agent = sms_agent or SMSAgent()
        self.email_agent = email_agent or EmailAgent()

        self.neo4j_driver = neo4j_driver
        if self.neo4j_driver is None:
            self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
            self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
            self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password")
            self.neo4j_driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
            )

        self.slack_webhook = os.getenv("SLACK_WEBHOOK_URL")

    async def send_dual_otp(self, txn: Dict[str, Any]) -> Dict[str, Any]:
        txn_id = txn["transaction_id"]
        codes = self.otp_manager.generate_otp(txn_id)

        sms_task = self.sms_agent.send_otp_async(txn["phone"], codes["sms"], txn)
        email_task = self.email_agent.send_otp_async(txn["email"], codes["email"], txn)

        results = await asyncio.gather(sms_task, email_task, return_exceptions=True)

        response = {"txn_id": txn_id, "sms_sent": False, "email_sent": False, "errors": []}
        channels = ["sms", "email"]
        for channel, result in zip(channels, results):
            if isinstance(result, Exception):
                response["errors"].append({"channel": channel, "error": str(result)})
            else:
                response[f"{channel}_sent"] = True
                self.otp_manager.mark_delivered(txn_id, channel)

        return response

    async def verify_sms(self, txn_id: str, code: str, txn: Dict[str, Any]) -> Dict[str, Any]:
        timestamp = self._current_timestamp()
        result = self.otp_manager.verify_sms(txn_id, code, timestamp)
        await self._apply_defenses(txn_id, txn, "sms", result)
        return result

    async def verify_email(self, txn_id: str, code: str, txn: Dict[str, Any]) -> Dict[str, Any]:
        timestamp = self._current_timestamp()
        result = self.otp_manager.verify_email(txn_id, code, timestamp)
        await self._apply_defenses(txn_id, txn, "email", result)
        return result

    def get_status(self, txn_id: str) -> Dict[str, Any]:
        return self.otp_manager.get_status(txn_id)

    def is_frozen(self, txn_id: str) -> bool:
        return self.otp_manager.redis_client.get(f"otp:{txn_id}:frozen") == "1"

    async def _apply_defenses(self, txn_id: str, txn: Dict[str, Any], channel: str, result: Dict[str, Any]):
        if not result.get("success"):
            if result.get("status") == "FAILED":
                await self.escalate(txn_id, txn, "otp_max_attempts_failed")
            return

        if channel == "sms" and self._is_recent_sim_swap(txn.get("user_id")):
            await self._block_due_to_sim_swap(txn_id, txn)
            return

        if self._device_mismatch(txn):
            await self._flag_suspicious(txn_id, txn, "device_mismatch")

        if self._fast_dual_confirmation(txn_id):
            await self._flag_suspicious(txn_id, txn, "fast_dual_confirmation")

        score = float(txn.get("score", 0.0))
        if score > 0.85:
            await self._send_mock_out_of_band(txn_id, txn)

    def _current_timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _device_mismatch(self, txn: Dict[str, Any]) -> bool:
        declared_device = txn.get("device_id")
        verification_device = txn.get("verification_device_id")
        if not declared_device or not verification_device:
            return False
        return declared_device != verification_device

    def _fast_dual_confirmation(self, txn_id: str) -> bool:
        status = self.otp_manager.get_status(txn_id)
        sms_time = status.get("sms_confirmed_at")
        email_time = status.get("email_confirmed_at")
        if not sms_time or not email_time:
            return False
        sms_dt = self._parse_timestamp(sms_time)
        email_dt = self._parse_timestamp(email_time)
        if not sms_dt or not email_dt:
            return False
        return abs((sms_dt - email_dt).total_seconds()) < 30

    def _parse_timestamp(self, value: str) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None

    async def _block_due_to_sim_swap(self, txn_id: str, txn: Dict[str, Any]):
        await self.escalate(txn_id, txn, "sim_swap_detected")

    async def _flag_suspicious(self, txn_id: str, txn: Dict[str, Any], reason: str):
        self._log_event(txn_id, "suspicious", reason)

    async def _send_mock_out_of_band(self, txn_id: str, txn: Dict[str, Any]):
        self._log_event(txn_id, "out_of_band", "push_notification_sent")

    def _log_event(self, txn_id: str, event_type: str, detail: str):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment("otp_interlock")
        with mlflow.start_run(run_name=f"otp_{txn_id}"):
            mlflow.log_param("txn_id", txn_id)
            mlflow.log_param("event_type", event_type)
            mlflow.log_param("detail", detail)

    def _freeze_transaction(self, txn_id: str):
        freeze_key = f"otp:{txn_id}:frozen"
        self.otp_manager.redis_client.set(freeze_key, "1", ex=self.otp_manager.ttl_seconds)

    async def escalate(self, txn_id: str, txn: Dict[str, Any], reason: str):
        self._freeze_transaction(txn_id)
        self._log_event(txn_id, "escalation", reason)
        await self._notify_slack(txn_id, txn, reason)

    async def _notify_slack(self, txn_id: str, txn: Dict[str, Any], reason: str):
        if not self.slack_webhook:
            return

        payload = json.dumps({
            "text": f"Escalation for transaction {txn_id}: {reason}"
        }).encode("utf-8")
        request = urllib.request.Request(
            self.slack_webhook,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            await asyncio.to_thread(urllib.request.urlopen, request, timeout=10)
        except urllib.error.URLError:
            self._log_event(txn_id, "escalation", "slack_notify_failed")

    def _is_recent_sim_swap(self, user_id: Optional[str]) -> bool:
        if not user_id:
            return False
        query = (
            "MATCH (u:User {user_id: $user_id}) "
            "RETURN u.sim_swap_timestamp AS sim_swap_timestamp"
        )
        with self.neo4j_driver.session() as session:
            result = session.run(query, user_id=user_id).single()
            if not result:
                return False
            timestamp = result.get("sim_swap_timestamp")
            if not timestamp:
                return False
            if isinstance(timestamp, str):
                timestamp = self._parse_timestamp(timestamp)
            if not isinstance(timestamp, datetime):
                return False
            return datetime.now(timezone.utc) - timestamp < timedelta(hours=72)
