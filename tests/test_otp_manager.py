import pytest
from pathlib import Path
import sys
import time
sys.path.append(str(Path(__file__).parent.parent))
from agents.otp_manager import OTPManager


@pytest.fixture
def otp_manager():
    manager = OTPManager()
    yield manager
    manager.reset("test-txn")


def test_generate_and_idempotent_codes(otp_manager):
    otp_manager.reset("test-txn")
    codes_first = otp_manager.generate_otp("test-txn")
    codes_second = otp_manager.generate_otp("test-txn")

    assert codes_first == codes_second
    assert len(codes_first["sms"]) == 6
    assert len(codes_first["email"]) == 6
    assert codes_first["sms"] != codes_first["email"]


def test_verify_sms_and_email(otp_manager):
    otp_manager.reset("test-txn")
    codes = otp_manager.generate_otp("test-txn")

    sms_result = otp_manager.verify_sms("test-txn", codes["sms"], "2026-05-08T12:00:00Z")
    assert sms_result["success"] is True
    assert sms_result["status"] in {"SMS_CONFIRMED", "APPROVED"}

    email_result = otp_manager.verify_email("test-txn", codes["email"], "2026-05-08T12:00:10Z")
    assert email_result["success"] is True
    assert email_result["status"] == "APPROVED"


def test_max_attempts_blocks(otp_manager):
    otp_manager.reset("test-txn")
    codes = otp_manager.generate_otp("test-txn")

    for i in range(1, 4):
        result = otp_manager.verify_sms("test-txn", "000000", "2026-05-08T12:00:00Z")
        if i < 3:
            assert result["success"] is False
            assert result["status"] == "PENDING_DUAL"
        else:
            assert result["success"] is False
            assert result["status"] == "FAILED"
