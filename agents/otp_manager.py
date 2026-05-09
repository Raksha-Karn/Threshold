import secrets
from pathlib import Path
from typing import Dict, Optional

import redis


class OTPManager:
    DEFAULT_TTL = 300
    MAX_ATTEMPTS = 3

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        ttl_seconds: int = DEFAULT_TTL,
        redis_client: Optional[redis.Redis] = None,
    ):
        self.redis_client = redis_client or redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
        self.ttl_seconds = ttl_seconds

    def _code_key(self, txn_id: str, channel: str) -> str:
        return f"otp:{txn_id}:{channel}"

    def _state_key(self, txn_id: str) -> str:
        return f"otp:{txn_id}:state"

    def _attempts_key(self, txn_id: str) -> str:
        return f"otp:{txn_id}:attempts"

    def _confirmed_at_key(self, txn_id: str, channel: str) -> str:
        return f"otp:{txn_id}:{channel}_confirmed_at"

    def _deliver_flag_key(self, txn_id: str, channel: str) -> str:
        return f"otp:{txn_id}:{channel}_delivered"

    def _status(self, txn_id: str) -> str:
        status = self.redis_client.get(self._state_key(txn_id))
        return status or "PENDING_DUAL"

    def _ensure_attempts_ttl(self, txn_id: str):
        attempts_key = self._attempts_key(txn_id)
        if self.redis_client.exists(attempts_key) and self.redis_client.ttl(attempts_key) == -1:
            self.redis_client.expire(attempts_key, self.ttl_seconds)

    def _get_or_create_code(self, txn_id: str, channel: str) -> str:
        key = self._code_key(txn_id, channel)
        existing = self.redis_client.get(key)
        if existing:
            return existing

        code = "".join(secrets.choice("0123456789") for _ in range(6))
        self.redis_client.set(key, code, ex=self.ttl_seconds)
        return code

    def generate_otp(self, txn_id: str) -> Dict[str, str]:
        sms_code = self._get_or_create_code(txn_id, "sms")
        email_code = self._get_or_create_code(txn_id, "email")
        state_key = self._state_key(txn_id)
        self.redis_client.set(state_key, "PENDING_DUAL", ex=self.ttl_seconds)
        self._ensure_attempts_ttl(txn_id)
        return {"sms": sms_code, "email": email_code}

    def get_status(self, txn_id: str) -> Dict[str, Optional[str]]:
        pipeline = self.redis_client.pipeline()
        pipeline.get(self._state_key(txn_id))
        pipeline.get(self._attempts_key(txn_id))
        pipeline.get(self._deliver_flag_key(txn_id, "sms"))
        pipeline.get(self._deliver_flag_key(txn_id, "email"))
        pipeline.get(self._confirmed_at_key(txn_id, "sms"))
        pipeline.get(self._confirmed_at_key(txn_id, "email"))
        state, attempts, sms_delivered, email_delivered, sms_confirmed_at, email_confirmed_at = pipeline.execute()

        return {
            "txn_id": txn_id,
            "status": state or "PENDING_DUAL",
            "attempts": attempts or "0",
            "sms_delivered": sms_delivered or "0",
            "email_delivered": email_delivered or "0",
            "sms_confirmed_at": sms_confirmed_at,
            "email_confirmed_at": email_confirmed_at,
        }

    def _increment_attempts(self, txn_id: str) -> int:
        attempts_key = self._attempts_key(txn_id)
        attempts = self.redis_client.incr(attempts_key)
        if attempts == 1:
            self.redis_client.expire(attempts_key, self.ttl_seconds)
        return attempts

    def _mark_failed(self, txn_id: str):
        state_key = self._state_key(txn_id)
        self.redis_client.set(state_key, "FAILED", ex=self.ttl_seconds)

    def _mark_state(self, txn_id: str, state: str):
        state_key = self._state_key(txn_id)
        self.redis_client.set(state_key, state, ex=self.ttl_seconds)

    def _set_confirmed_at(self, txn_id: str, channel: str, timestamp: str):
        key = self._confirmed_at_key(txn_id, channel)
        self.redis_client.set(key, timestamp, ex=self.ttl_seconds)

    def _transition_state(self, txn_id: str, channel: str) -> str:
        current_state = self._status(txn_id)
        completed = self.redis_client.exists(self._confirmed_at_key(txn_id, "sms")) and self.redis_client.exists(self._confirmed_at_key(txn_id, "email"))

        if completed:
            new_state = "APPROVED"
        elif channel == "sms":
            new_state = "SMS_CONFIRMED"
        else:
            new_state = "EMAIL_CONFIRMED"

        self._mark_state(txn_id, new_state)
        return new_state

    def _validate_code(self, txn_id: str, code: str, channel: str) -> bool:
        expected = self.redis_client.get(self._code_key(txn_id, channel))
        return expected is not None and expected == str(code)

    def verify_channel(self, txn_id: str, code: str, channel: str, timestamp: str) -> Dict[str, object]:
        state = self._status(txn_id)
        if state in {"FAILED", "APPROVED"}:
            return {"success": False, "status": state, "message": "OTP process completed"}

        if not self._validate_code(txn_id, code, channel):
            attempts = self._increment_attempts(txn_id)
            if attempts >= self.MAX_ATTEMPTS:
                self._mark_failed(txn_id)
                return {"success": False, "status": "FAILED", "message": "Max attempts exceeded"}
            return {"success": False, "status": state, "message": "Invalid code", "attempts": attempts}

        self._set_confirmed_at(txn_id, channel, timestamp)
        new_state = self._transition_state(txn_id, channel)
        return {"success": True, "status": new_state, "channel": channel}

    def verify_sms(self, txn_id: str, code: str, timestamp: str) -> Dict[str, object]:
        return self.verify_channel(txn_id, code, "sms", timestamp)

    def verify_email(self, txn_id: str, code: str, timestamp: str) -> Dict[str, object]:
        return self.verify_channel(txn_id, code, "email", timestamp)

    def mark_delivered(self, txn_id: str, channel: str):
        key = self._deliver_flag_key(txn_id, channel)
        self.redis_client.set(key, "1", ex=self.ttl_seconds)

    def reset(self, txn_id: str):
        keys = [
            self._code_key(txn_id, "sms"),
            self._code_key(txn_id, "email"),
            self._state_key(txn_id),
            self._attempts_key(txn_id),
            self._confirmed_at_key(txn_id, "sms"),
            self._confirmed_at_key(txn_id, "email"),
            self._deliver_flag_key(txn_id, "sms"),
            self._deliver_flag_key(txn_id, "email"),
        ]
        self.redis_client.delete(*keys)
