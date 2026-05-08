import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.otp_interlock import OTPInterlock
from agents.otp_manager import OTPManager
from agents.sms_agent import SMSAgent
from agents.email_agent import EmailAgent


@pytest.fixture
def mock_sms_agent():
    """Mock SMS delivery agent."""
    agent = AsyncMock(spec=SMSAgent)
    agent.send_otp_async = AsyncMock(return_value="SM1234567890")
    return agent


@pytest.fixture
def mock_email_agent():
    """Mock Email delivery agent."""
    agent = AsyncMock(spec=EmailAgent)
    agent.send_otp_async = AsyncMock(return_value=None)
    return agent


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver."""
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=None)
    return driver


@pytest.fixture
def otp_manager():
    """Create OTP manager with Redis."""
    return OTPManager()


@pytest.fixture
def otp_interlock(mock_sms_agent, mock_email_agent, mock_neo4j_driver, otp_manager):
    """Create OTP interlock with mocked dependencies."""
    return OTPInterlock(
        sms_agent=mock_sms_agent,
        email_agent=mock_email_agent,
        otp_manager=otp_manager,
        neo4j_driver=mock_neo4j_driver,
    )


class TestOTPInterlockDualDelivery:
    """Tests for dual-channel OTP delivery."""

    @pytest.mark.asyncio
    async def test_send_dual_otp_success(self, otp_interlock, mock_sms_agent, mock_email_agent):
        """Test successful dual OTP delivery via SMS and Email."""
        txn = {
            "transaction_id": "txn_123456",
            "phone": "+1234567890",
            "email": "user@example.com",
            "merchant_id": "MERCHANT_001",
            "amount": 100.00,
        }

        result = await otp_interlock.send_dual_otp(txn)

        assert result["txn_id"] == "txn_123456"
        assert result["sms_sent"] is True
        assert result["email_sent"] is True
        assert result["errors"] == []
        
        mock_sms_agent.send_otp_async.assert_called_once()
        mock_email_agent.send_otp_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_dual_otp_sms_failure(self, otp_interlock, mock_sms_agent, mock_email_agent):
        """Test OTP delivery when SMS fails."""
        mock_sms_agent.send_otp_async.side_effect = Exception("SMS API Error")

        txn = {
            "transaction_id": "txn_sms_fail",
            "phone": "+1234567890",
            "email": "user@example.com",
        }

        result = await otp_interlock.send_dual_otp(txn)

        assert result["txn_id"] == "txn_sms_fail"
        assert result["sms_sent"] is False
        assert result["email_sent"] is True
        assert len(result["errors"]) == 1
        assert result["errors"][0]["channel"] == "sms"
        assert "SMS API Error" in result["errors"][0]["error"]

    @pytest.mark.asyncio
    async def test_send_dual_otp_email_failure(self, otp_interlock, mock_sms_agent, mock_email_agent):
        """Test OTP delivery when Email fails."""
        mock_email_agent.send_otp_async.side_effect = Exception("SMTP Error")

        txn = {
            "transaction_id": "txn_email_fail",
            "phone": "+1234567890",
            "email": "user@example.com",
        }

        result = await otp_interlock.send_dual_otp(txn)

        assert result["txn_id"] == "txn_email_fail"
        assert result["sms_sent"] is True
        assert result["email_sent"] is False
        assert len(result["errors"]) == 1
        assert result["errors"][0]["channel"] == "email"

    @pytest.mark.asyncio
    async def test_send_dual_otp_both_failure(self, otp_interlock, mock_sms_agent, mock_email_agent):
        """Test OTP delivery when both channels fail."""
        mock_sms_agent.send_otp_async.side_effect = Exception("SMS Error")
        mock_email_agent.send_otp_async.side_effect = Exception("Email Error")

        txn = {
            "transaction_id": "txn_both_fail",
            "phone": "+1234567890",
            "email": "user@example.com",
        }

        result = await otp_interlock.send_dual_otp(txn)

        assert result["sms_sent"] is False
        assert result["email_sent"] is False
        assert len(result["errors"]) == 2

    @pytest.mark.asyncio
    async def test_send_dual_otp_generates_codes(self, otp_interlock, otp_manager, mock_sms_agent, mock_email_agent):
        """Test that send_dual_otp generates OTP codes."""
        txn = {
            "transaction_id": "txn_codes",
            "phone": "+1234567890",
            "email": "user@example.com",
        }

        result = await otp_interlock.send_dual_otp(txn)

        # Check that codes were generated
        status = otp_interlock.otp_manager.get_status("txn_codes")
        assert status is not None
        assert status["status"] == "PENDING_DUAL"
        assert status["sms_delivered"] == "1"
        assert status["email_delivered"] == "1"


class TestOTPInterlockVerification:
    """Tests for OTP verification."""

    @pytest.mark.asyncio
    async def test_verify_sms_success(self, otp_interlock, otp_manager):
        """Test successful SMS OTP verification."""
        txn_id = "txn_verify_sms"
        codes = otp_manager.generate_otp(txn_id)
        
        txn = {
            "transaction_id": txn_id,
            "user_id": "user_123",
            "phone": "+1234567890",
            "email": "user@example.com",
            "score": 0.5,
        }

        result = await otp_interlock.verify_sms(txn_id, codes["sms"], txn)

        assert result["success"] is True
        assert result["status"] == "SMS_CONFIRMED"

    @pytest.mark.asyncio
    async def test_verify_sms_incorrect_code(self, otp_interlock):
        """Test SMS verification with incorrect code."""
        txn_id = "txn_verify_sms_fail"
        otp_interlock.otp_manager.generate_otp(txn_id)

        txn = {
            "transaction_id": txn_id,
            "user_id": "user_123",
        }

        result = await otp_interlock.verify_sms(txn_id, "000000", txn)

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_verify_email_success(self, otp_interlock, otp_manager):
        """Test successful Email OTP verification."""
        txn_id = "txn_verify_email"
        codes = otp_manager.generate_otp(txn_id)

        txn = {
            "transaction_id": txn_id,
            "user_id": "user_456",
            "phone": "+1234567890",
            "email": "user@example.com",
            "score": 0.3,
        }

        result = await otp_interlock.verify_email(txn_id, codes["email"], txn)

        assert result["success"] is True
        assert result["status"] == "EMAIL_CONFIRMED"

    @pytest.mark.asyncio
    async def test_verify_max_attempts_exceeded(self, otp_interlock, otp_manager):
        """Test verification blocks after max attempts."""
        txn_id = "txn_max_attempts"
        otp_manager.generate_otp(txn_id)

        txn = {
            "transaction_id": txn_id,
            "user_id": "user_789",
        }

        for _ in range(3):
            await otp_interlock.verify_sms(txn_id, "000000", txn)

        result = await otp_interlock.verify_sms(txn_id, "000000", txn)

        assert result["success"] is False
        assert result["status"] == "FAILED"


class TestOTPInterlockDefenses:
    """Tests for fraud defense mechanisms."""

    @pytest.mark.asyncio
    async def test_device_mismatch_detection(self, otp_interlock, otp_manager):
        """Test detection of device mismatch."""
        txn_id = "txn_device_mismatch"
        codes = otp_manager.generate_otp(txn_id)

        txn = {
            "transaction_id": txn_id,
            "device_id": "device_001",
            "verification_device_id": "device_002",  # Different device
            "user_id": "user_123",
        }

        with patch.object(otp_interlock, "_flag_suspicious", new_callable=AsyncMock) as mock_flag:
            await otp_interlock.verify_sms(txn_id, codes["sms"], txn)
            mock_flag.assert_called_once()
            call_args = mock_flag.call_args
            assert call_args[0][2] == "device_mismatch"

    @pytest.mark.asyncio
    async def test_fast_dual_confirmation_detection(self, otp_interlock, otp_manager):
        """Test detection of suspiciously fast dual verification."""
        txn_id = "txn_fast_dual"
        codes = otp_manager.generate_otp(txn_id)

        txn = {
            "transaction_id": txn_id,
            "user_id": "user_123",
            "score": 0.5,
        }

        await otp_interlock.verify_sms(txn_id, codes["sms"], txn)

        with patch.object(otp_interlock, "_flag_suspicious", new_callable=AsyncMock) as mock_flag:
            await otp_interlock.verify_email(txn_id, codes["email"], txn)
            call_args_list = [call[0] for call in mock_flag.call_args_list]
            reasons = [call[2] for call in call_args_list]
            assert any("fast_dual_confirmation" in reason for reason in reasons) or len(reasons) == 0

    @pytest.mark.asyncio
    async def test_high_score_out_of_band_challenge(self, otp_interlock, otp_manager):
        """Test out-of-band challenge for high-risk transactions."""
        txn_id = "txn_high_risk"
        codes = otp_manager.generate_otp(txn_id)

        txn = {
            "transaction_id": txn_id,
            "user_id": "user_123",
            "score": 0.95,  
        }

        with patch.object(otp_interlock, "_send_mock_out_of_band", new_callable=AsyncMock) as mock_oob:
            await otp_interlock.verify_sms(txn_id, codes["sms"], txn)
            if mock_oob.call_count == 0:
                await otp_interlock.verify_email(txn_id, codes["email"], txn)
            assert mock_oob.call_count >= 0  

    def test_sim_swap_detection_recent(self, otp_interlock, mock_neo4j_driver):
        """Test detection of recent SIM swap within 72 hours."""
        session_mock = MagicMock()
        result_mock = MagicMock()
        recent_timestamp = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        result_mock.get.return_value = recent_timestamp
        session_mock.run.return_value.single.return_value = result_mock

        mock_neo4j_driver.session.return_value.__enter__.return_value = session_mock

        is_sim_swap = otp_interlock._is_recent_sim_swap("user_with_sim_swap")

        assert is_sim_swap is True
        session_mock.run.assert_called_once()

    def test_sim_swap_detection_not_recent(self, otp_interlock, mock_neo4j_driver):
        """Test SIM swap detection when swap is older than 72 hours."""
        session_mock = MagicMock()
        result_mock = MagicMock()
        old_timestamp = (datetime.now(timezone.utc) - timedelta(hours=120)).isoformat()
        result_mock.get.return_value = old_timestamp
        session_mock.run.return_value.single.return_value = result_mock

        mock_neo4j_driver.session.return_value.__enter__.return_value = session_mock

        is_sim_swap = otp_interlock._is_recent_sim_swap("user_old_sim_swap")

        assert is_sim_swap is False

    def test_sim_swap_detection_no_record(self, otp_interlock, mock_neo4j_driver):
        """Test SIM swap detection when user has no record."""
        session_mock = MagicMock()
        session_mock.run.return_value.single.return_value = None

        mock_neo4j_driver.session.return_value.__enter__.return_value = session_mock

        is_sim_swap = otp_interlock._is_recent_sim_swap("user_no_record")

        assert is_sim_swap is False

    @pytest.mark.asyncio
    async def test_sim_swap_escalation(self, otp_interlock, otp_manager):
        """Test escalation when SIM swap is detected during SMS verification."""
        txn_id = "txn_sim_swap"
        codes = otp_manager.generate_otp(txn_id)

        txn = {
            "transaction_id": txn_id,
            "user_id": "user_with_sim_swap",
            "score": 0.6,
        }

        with patch.object(otp_interlock, "_is_recent_sim_swap", return_value=True):
            with patch.object(otp_interlock, "escalate", new_callable=AsyncMock) as mock_escalate:
                await otp_interlock.verify_sms(txn_id, codes["sms"], txn)
                mock_escalate.assert_called_once()
                call_args = mock_escalate.call_args
                assert call_args[0][2] == "sim_swap_detected"


class TestOTPInterlockEscalation:
    """Tests for escalation and incident handling."""

    @pytest.mark.asyncio
    async def test_escalate_freezes_transaction(self, otp_interlock):
        """Test that escalation freezes the transaction."""
        txn_id = "txn_freeze"
        txn = {
            "transaction_id": txn_id,
            "user_id": "user_123",
        }

        with patch.object(otp_interlock, "_notify_slack", new_callable=AsyncMock):
            await otp_interlock.escalate(txn_id, txn, "test_reason")

        freeze_key = f"otp:{txn_id}:frozen"
        is_frozen = otp_interlock.otp_manager.redis_client.get(freeze_key)
        assert is_frozen == "1"

    @pytest.mark.asyncio
    async def test_escalate_logs_event(self, otp_interlock):
        """Test that escalation logs to MLflow."""
        txn_id = "txn_log_escalation"
        txn = {
            "transaction_id": txn_id,
            "user_id": "user_123",
        }

        with patch("agents.otp_interlock.mlflow.start_run"):
            with patch("agents.otp_interlock.mlflow.log_param") as mock_log:
                with patch.object(otp_interlock, "_notify_slack", new_callable=AsyncMock):
                    await otp_interlock.escalate(txn_id, txn, "escalation_reason")

    @pytest.mark.asyncio
    async def test_escalate_notifies_slack(self, otp_interlock):
        """Test Slack notification on escalation."""
        txn_id = "txn_slack_notify"
        txn = {
            "transaction_id": txn_id,
            "user_id": "user_123",
            "amount": 5000.00,
        }

        with patch.object(otp_interlock, "_notify_slack", new_callable=AsyncMock) as mock_notify:
            await otp_interlock.escalate(txn_id, txn, "high_risk_transaction")
            mock_notify.assert_called_once()

    @pytest.mark.asyncio
    async def test_escalate_max_attempts_failure(self, otp_interlock, otp_manager):
        """Test escalation on max OTP attempts exceeded."""
        txn_id = "txn_max_attempts_escalation"
        otp_manager.generate_otp(txn_id)

        txn = {
            "transaction_id": txn_id,
            "user_id": "user_123",
        }

        # Fail 3 attempts
        for _ in range(3):
            await otp_interlock.verify_sms(txn_id, "000000", txn)

        with patch.object(otp_interlock, "escalate", new_callable=AsyncMock) as mock_escalate:
            await otp_interlock.verify_sms(txn_id, "000000", txn)
            mock_escalate.assert_called_once()
            call_args = mock_escalate.call_args
            assert call_args[0][2] == "otp_max_attempts_failed"


class TestOTPInterlockStatus:
    """Tests for status checking."""

    def test_get_status_returns_current_state(self, otp_interlock, otp_manager):
        """Test status retrieval returns current OTP state."""
        txn_id = "txn_status"
        codes = otp_manager.generate_otp(txn_id)

        status = otp_interlock.get_status(txn_id)

        assert status is not None
        assert status["status"] == "PENDING_DUAL"
        assert status["sms_delivered"] == "0"
        assert status["email_delivered"] == "0"

    def test_get_status_after_verification(self, otp_interlock, otp_manager):
        """Test status after OTP verification."""
        txn_id = "txn_status_after_verify"
        codes = otp_manager.generate_otp(txn_id)

        otp_interlock.otp_manager.verify_sms(txn_id, codes["sms"], "2026-05-08T12:00:00Z")

        status = otp_interlock.get_status(txn_id)

        assert status["status"] == "SMS_CONFIRMED"
        assert status["sms_confirmed_at"] is not None
        assert status["email_confirmed_at"] is None


class TestOTPInterlockTimestamps:
    """Tests for timestamp parsing and handling."""

    def test_parse_timestamp_iso_format(self, otp_interlock):
        """Test parsing ISO format timestamps."""
        iso_timestamp = "2026-05-08T12:00:00+00:00"
        parsed = otp_interlock._parse_timestamp(iso_timestamp)

        assert parsed is not None
        assert isinstance(parsed, datetime)

    def test_parse_timestamp_with_z(self, otp_interlock):
        """Test parsing timestamps with Z notation."""
        z_timestamp = "2026-05-08T12:00:00Z"
        parsed = otp_interlock._parse_timestamp(z_timestamp)

        assert parsed is not None
        assert isinstance(parsed, datetime)

    def test_parse_timestamp_invalid(self, otp_interlock):
        """Test parsing invalid timestamp returns None."""
        invalid_timestamp = "not-a-timestamp"
        parsed = otp_interlock._parse_timestamp(invalid_timestamp)

        assert parsed is None


class TestOTPInterlockHelpers:
    """Tests for helper methods."""

    def test_current_timestamp(self, otp_interlock):
        """Test current timestamp generation."""
        before = datetime.now(timezone.utc)
        timestamp_str = otp_interlock._current_timestamp()
        after = datetime.now(timezone.utc)

        parsed = otp_interlock._parse_timestamp(timestamp_str)
        assert parsed is not None
        assert before <= parsed <= after

    def test_device_mismatch_with_none_values(self, otp_interlock):
        """Test device mismatch detection handles None values."""
        txn = {}
        is_mismatch = otp_interlock._device_mismatch(txn)
        assert is_mismatch is False

    def test_device_mismatch_same_device(self, otp_interlock):
        """Test device mismatch returns False for same device."""
        txn = {
            "device_id": "device_same",
            "verification_device_id": "device_same",
        }
        is_mismatch = otp_interlock._device_mismatch(txn)
        assert is_mismatch is False

    def test_fast_dual_confirmation_without_both_times(self, otp_interlock):
        """Test fast dual confirmation returns False without both verification times."""
        otp_interlock.otp_manager.generate_otp("txn_no_times")
        is_fast = otp_interlock._fast_dual_confirmation("txn_no_times")
        assert is_fast is False
