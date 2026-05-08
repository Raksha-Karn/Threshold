import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.sms_agent import SMSAgent


@pytest.fixture
def mock_twilio_client():
    """Mock Twilio client for testing."""
    with patch("agents.sms_agent.Client") as mock_client_class:
        mock_instance = MagicMock()
        mock_client_class.return_value = mock_instance

        # Mock message creation
        mock_message = Mock()
        mock_message.sid = "SM1234567890abcdef1234567890abcdef"
        mock_instance.messages.create.return_value = mock_message

        yield mock_instance


@pytest.fixture
def sms_agent(mock_twilio_client):
    """Create an SMS agent with mocked Twilio client."""
    with patch.dict(
        "os.environ",
        {
            "TWILIO_ACCOUNT_SID": "ACtest123",
            "TWILIO_AUTH_TOKEN": "token123",
            "TWILIO_FROM_NUMBER": "+1234567890",
        },
    ):
        agent = SMSAgent()
        agent.client = mock_twilio_client
        return agent


class TestSMSAgent:
    def test_sms_agent_init_with_env_vars(self):
        """Test SMSAgent initialization with environment variables."""
        with patch.dict(
            "os.environ",
            {
                "TWILIO_ACCOUNT_SID": "ACtest123",
                "TWILIO_AUTH_TOKEN": "token123",
                "TWILIO_FROM_NUMBER": "+1234567890",
            },
        ):
            with patch("agents.sms_agent.Client"):
                agent = SMSAgent()
                assert agent.account_sid == "ACtest123"
                assert agent.auth_token == "token123"
                assert agent.from_number == "+1234567890"

    def test_sms_agent_init_with_explicit_params(self):
        """Test SMSAgent initialization with explicit parameters."""
        with patch("agents.sms_agent.Client"):
            agent = SMSAgent(
                account_sid="ACexplicit",
                auth_token="tokenexplicit",
                from_number="+9876543210",
            )
            assert agent.account_sid == "ACexplicit"
            assert agent.auth_token == "tokenexplicit"
            assert agent.from_number == "+9876543210"

    def test_sms_agent_init_missing_credentials(self):
        """Test SMSAgent raises error when credentials are missing."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("agents.sms_agent.Client"):
                with pytest.raises(ValueError, match="Twilio credentials"):
                    SMSAgent()

    def test_send_otp_basic(self, sms_agent, mock_twilio_client):
        """Test basic OTP SMS sending."""
        sms_id = sms_agent.send_otp("+1111111111", "123456")

        assert sms_id == "SM1234567890abcdef1234567890abcdef"
        mock_twilio_client.messages.create.assert_called_once()

        call_args = mock_twilio_client.messages.create.call_args
        assert call_args.kwargs["to"] == "+1111111111"
        assert "123456" in call_args.kwargs["body"]
        assert "5 minutes" in call_args.kwargs["body"]
        assert call_args.kwargs["from_"] == "+1234567890"

    def test_send_otp_with_transaction_details(self, sms_agent, mock_twilio_client):
        """Test OTP SMS sending with transaction details."""
        txn_details = {
            "merchant_id": "MERCHANT123",
            "amount": 500.50,
        }

        sms_id = sms_agent.send_otp("+1111111111", "654321", txn_details)

        assert sms_id == "SM1234567890abcdef1234567890abcdef"
        call_args = mock_twilio_client.messages.create.call_args
        body = call_args.kwargs["body"]

        assert "654321" in body
        assert "500.5" in body
        assert "MERCHANT123" in body

    def test_send_otp_with_merchant_type(self, sms_agent, mock_twilio_client):
        """Test OTP SMS sending with merchant type instead of merchant ID."""
        txn_details = {
            "merchant_type": "ONLINE_RETAIL",
            "amount": 1200.00,
        }

        sms_id = sms_agent.send_otp("+2222222222", "789012", txn_details)

        assert sms_id == "SM1234567890abcdef1234567890abcdef"
        call_args = mock_twilio_client.messages.create.call_args
        body = call_args.kwargs["body"]

        assert "789012" in body
        assert "1200" in body
        assert "ONLINE_RETAIL" in body

    def test_send_otp_with_partial_transaction_details(self, sms_agent, mock_twilio_client):
        """Test OTP SMS sending with only amount in transaction details."""
        txn_details = {
            "amount": 75.25,
        }

        sms_id = sms_agent.send_otp("+3333333333", "111111", txn_details)

        assert sms_id == "SM1234567890abcdef1234567890abcdef"
        call_args = mock_twilio_client.messages.create.call_args
        body = call_args.kwargs["body"]

        assert "111111" in body
        assert "75.25" in body

    def test_send_otp_with_none_details(self, sms_agent, mock_twilio_client):
        """Test OTP SMS sending with None transaction details."""
        sms_id = sms_agent.send_otp("+4444444444", "222222", None)

        assert sms_id == "SM1234567890abcdef1234567890abcdef"
        call_args = mock_twilio_client.messages.create.call_args
        body = call_args.kwargs["body"]

        assert "222222" in body
        assert "5 minutes" in body

    @pytest.mark.asyncio
    async def test_send_otp_async_basic(self, sms_agent, mock_twilio_client):
        """Test async OTP SMS sending."""
        sms_id = await sms_agent.send_otp_async("+5555555555", "333333")

        assert sms_id == "SM1234567890abcdef1234567890abcdef"
        mock_twilio_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_otp_async_with_details(self, sms_agent, mock_twilio_client):
        """Test async OTP SMS sending with transaction details."""
        txn_details = {
            "merchant_id": "MERCHANT_ASYNC",
            "amount": 999.99,
        }

        sms_id = await sms_agent.send_otp_async("+6666666666", "444444", txn_details)

        assert sms_id == "SM1234567890abcdef1234567890abcdef"
        call_args = mock_twilio_client.messages.create.call_args
        body = call_args.kwargs["body"]

        assert "444444" in body
        assert "MERCHANT_ASYNC" in body
        assert "999.99" in body

    @pytest.mark.asyncio
    async def test_send_otp_async_multiple_concurrent(self, sms_agent, mock_twilio_client):
        """Test concurrent async OTP SMS sends."""
        sms_ids = await asyncio.gather(
            sms_agent.send_otp_async("+7777777777", "555555"),
            sms_agent.send_otp_async("+8888888888", "666666"),
            sms_agent.send_otp_async("+9999999999", "777777"),
        )

        assert len(sms_ids) == 3
        assert all(sid == "SM1234567890abcdef1234567890abcdef" for sid in sms_ids)
        assert mock_twilio_client.messages.create.call_count == 3

    def test_send_otp_different_phone_formats(self, sms_agent, mock_twilio_client):
        """Test OTP SMS sending with different phone number formats."""
        phone_numbers = [
            "+1234567890",
            "+44-123-456-7890",
            "+91-9999-999999",
        ]

        for phone in phone_numbers:
            sms_agent.send_otp(phone, "123456")
            call_args = mock_twilio_client.messages.create.call_args
            assert call_args.kwargs["to"] == phone

    def test_send_otp_twilio_api_error(self, sms_agent, mock_twilio_client):
        """Test OTP SMS sending handles Twilio API errors."""
        mock_twilio_client.messages.create.side_effect = Exception("Twilio API Error")

        with pytest.raises(Exception, match="Twilio API Error"):
            sms_agent.send_otp("+1111111111", "123456")
