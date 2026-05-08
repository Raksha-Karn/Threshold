import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open, call

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.email_agent import EmailAgent


@pytest.fixture
def mock_smtp():
    """Mock SMTP connection for testing."""
    with patch("agents.email_agent.smtplib.SMTP") as mock_smtp_class:
        mock_instance = MagicMock()
        mock_smtp_class.return_value.__enter__.return_value = mock_instance

        yield mock_instance


@pytest.fixture
def email_agent(mock_smtp):
    """Create an Email agent with mocked SMTP."""
    with patch.dict(
        "os.environ",
        {
            "EMAIL_SMTP_HOST": "smtp.test.com",
            "EMAIL_SMTP_PORT": "587",
            "EMAIL_USERNAME": "test@test.com",
            "EMAIL_PASSWORD": "testpass",
            "EMAIL_FROM_ADDRESS": "test@test.com",
        },
    ):
        agent = EmailAgent()
        return agent


class TestEmailAgent:
    def test_email_agent_init_with_env_vars(self):
        """Test EmailAgent initialization with environment variables."""
        with patch.dict(
            "os.environ",
            {
                "EMAIL_SMTP_HOST": "smtp.test.com",
                "EMAIL_SMTP_PORT": "587",
                "EMAIL_USERNAME": "test@test.com",
                "EMAIL_PASSWORD": "testpass",
                "EMAIL_FROM_ADDRESS": "test@test.com",
            },
        ):
            agent = EmailAgent()
            assert agent.smtp_host == "smtp.test.com"
            assert agent.smtp_port == 587
            assert agent.username == "test@test.com"
            assert agent.password == "testpass"
            assert agent.from_address == "test@test.com"
            assert agent.use_tls is True

    def test_email_agent_init_with_explicit_params(self):
        """Test EmailAgent initialization with explicit parameters."""
        agent = EmailAgent(
            smtp_host="smtp.explicit.com",
            smtp_port=465,
            username="explicit@test.com",
            password="explicitpass",
            from_address="explicit@test.com",
            use_tls=False,
        )
        assert agent.smtp_host == "smtp.explicit.com"
        assert agent.smtp_port == 465
        assert agent.username == "explicit@test.com"
        assert agent.password == "explicitpass"
        assert agent.from_address == "explicit@test.com"
        assert agent.use_tls is False

    def test_email_agent_init_default_port(self):
        """Test EmailAgent initialization with default port."""
        with patch.dict(
            "os.environ",
            {
                "EMAIL_SMTP_HOST": "smtp.test.com",
                "EMAIL_USERNAME": "test@test.com",
                "EMAIL_PASSWORD": "testpass",
                "EMAIL_FROM_ADDRESS": "test@test.com",
            },
        ):
            agent = EmailAgent()
            assert agent.smtp_port == 587  # default value

    def test_email_agent_init_missing_credentials(self):
        """Test EmailAgent raises error when credentials are missing."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Email SMTP settings"):
                EmailAgent()

    def test_send_otp_basic(self, email_agent, mock_smtp):
        """Test basic OTP email sending."""
        email_agent.send_otp("recipient@test.com", "123456")

        mock_smtp.starttls.assert_called_once()
        mock_smtp.login.assert_called_once_with("test@test.com", "testpass")
        mock_smtp.send_message.assert_called_once()

        sent_message = mock_smtp.send_message.call_args[0][0]
        assert sent_message["Subject"] == "Your OTP Code for Transaction Verification"
        assert sent_message["From"] == "test@test.com"
        assert sent_message["To"] == "recipient@test.com"
        
        # Check plain text part
        payloads = sent_message.get_payload()
        assert len(payloads) == 2
        assert "123456" in str(sent_message)

    def test_send_otp_with_transaction_details(self, email_agent, mock_smtp):
        """Test OTP email sending with transaction details."""
        txn_details = {
            "merchant_id": "MERCHANT_EMAIL_123",
            "amount": 500.50,
        }

        email_agent.send_otp("recipient@test.com", "654321", txn_details)

        mock_smtp.send_message.assert_called_once()
        sent_message = mock_smtp.send_message.call_args[0][0]

        message_content = sent_message.get_payload()[1].get_payload()
        assert "654321" in message_content
        assert "500.5" in message_content or "500.50" in message_content
        assert "MERCHANT_EMAIL_123" in message_content

    def test_send_otp_with_merchant_type(self, email_agent, mock_smtp):
        """Test OTP email sending with merchant type."""
        txn_details = {
            "merchant_type": "ONLINE_RETAIL",
            "amount": 1200.00,
        }

        email_agent.send_otp("user@example.com", "789012", txn_details)

        mock_smtp.send_message.assert_called_once()
        sent_message = mock_smtp.send_message.call_args[0][0]
        message_content = sent_message.get_payload()[1].get_payload()

        assert "789012" in message_content
        assert "1200" in message_content
        assert "ONLINE_RETAIL" in message_content

    def test_send_otp_with_only_amount(self, email_agent, mock_smtp):
        """Test OTP email sending with only amount in transaction details."""
        txn_details = {
            "amount": 75.25,
        }

        email_agent.send_otp("customer@example.com", "111111", txn_details)

        mock_smtp.send_message.assert_called_once()
        sent_message = mock_smtp.send_message.call_args[0][0]
        message_content = sent_message.get_payload()[1].get_payload()

        assert "111111" in message_content
        assert "75.25" in message_content

    def test_send_otp_with_none_details(self, email_agent, mock_smtp):
        """Test OTP email sending with None transaction details."""
        email_agent.send_otp("contact@example.com", "222222", None)

        mock_smtp.send_message.assert_called_once()
        sent_message = mock_smtp.send_message.call_args[0][0]

        message_str = str(sent_message)
        assert "222222" in message_str
        assert "5 minutes" in message_str

    def test_send_otp_html_format(self, email_agent, mock_smtp):
        """Test OTP email is sent in HTML format."""
        email_agent.send_otp("test@example.com", "333333")

        mock_smtp.send_message.assert_called_once()
        sent_message = mock_smtp.send_message.call_args[0][0]

        # Check that HTML alternative is present
        payloads = sent_message.get_payload()
        assert len(payloads) == 2  # plain text and HTML
        assert payloads[1].get_content_type() == "text/html"

    def test_send_otp_html_with_details_formatting(self, email_agent, mock_smtp):
        """Test OTP email HTML formatting includes transaction details."""
        txn_details = {
            "merchant_id": "SHOP123",
            "amount": 100.00,
        }

        email_agent.send_otp("buyer@example.com", "444444", txn_details)

        sent_message = mock_smtp.send_message.call_args[0][0]
        payloads = sent_message.get_payload()
        html_content = payloads[1].get_payload()

        assert "<html>" in html_content
        assert "<strong>444444</strong>" in html_content
        assert "100" in html_content
        assert "SHOP123" in html_content

    def test_send_otp_no_tls(self, mock_smtp):
        """Test OTP email sending without TLS."""
        with patch.dict(
            "os.environ",
            {
                "EMAIL_SMTP_HOST": "smtp.test.com",
                "EMAIL_SMTP_PORT": "25",
                "EMAIL_USERNAME": "test@test.com",
                "EMAIL_PASSWORD": "testpass",
                "EMAIL_FROM_ADDRESS": "test@test.com",
            },
        ):
            agent = EmailAgent(use_tls=False)

        agent.send_otp("test@example.com", "555555")

        mock_smtp.starttls.assert_not_called()
        mock_smtp.login.assert_called_once()
        mock_smtp.send_message.assert_called_once()

    def test_send_otp_with_tls(self, email_agent, mock_smtp):
        """Test OTP email sending with TLS enabled (default)."""
        email_agent.send_otp("secure@example.com", "666666")

        mock_smtp.starttls.assert_called_once()
        mock_smtp.login.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_otp_async_basic(self, email_agent, mock_smtp):
        """Test async OTP email sending."""
        await email_agent.send_otp_async("async@example.com", "777777")

        mock_smtp.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_otp_async_with_details(self, email_agent, mock_smtp):
        """Test async OTP email sending with transaction details."""
        txn_details = {
            "merchant_id": "ASYNC_MERCHANT",
            "amount": 250.75,
        }

        await email_agent.send_otp_async("async_txn@example.com", "888888", txn_details)

        mock_smtp.send_message.assert_called_once()
        sent_message = mock_smtp.send_message.call_args[0][0]
        message_content = sent_message.get_payload()[1].get_payload()

        assert "888888" in message_content
        assert "ASYNC_MERCHANT" in message_content
        assert "250.75" in message_content or "250" in message_content

    @pytest.mark.asyncio
    async def test_send_otp_async_multiple_concurrent(self, email_agent, mock_smtp):
        """Test concurrent async OTP email sends."""
        await asyncio.gather(
            email_agent.send_otp_async("email1@example.com", "111111"),
            email_agent.send_otp_async("email2@example.com", "222222"),
            email_agent.send_otp_async("email3@example.com", "333333"),
        )

        assert mock_smtp.send_message.call_count == 3

    def test_send_otp_multiple_recipients(self, email_agent, mock_smtp):
        """Test OTP email sending to multiple recipients."""
        recipients = [
            "user1@example.com",
            "user2@example.com",
            "user3@example.com",
        ]

        for recipient in recipients:
            email_agent.send_otp(recipient, "123456")

        assert mock_smtp.send_message.call_count == 3

        for i, recipient in enumerate(recipients):
            call_args_list = mock_smtp.send_message.call_args_list
            sent_message = call_args_list[i][0][0]
            assert sent_message["To"] == recipient

    def test_send_otp_smtp_error_handling(self, email_agent, mock_smtp):
        """Test OTP email sending handles SMTP errors."""
        mock_smtp.send_message.side_effect = Exception("SMTP Error")

        with pytest.raises(Exception, match="SMTP Error"):
            email_agent.send_otp("test@example.com", "123456")

    def test_send_otp_login_error(self, email_agent, mock_smtp):
        """Test OTP email sending handles login errors."""
        mock_smtp.login.side_effect = Exception("Authentication failed")

        with pytest.raises(Exception, match="Authentication failed"):
            email_agent.send_otp("test@example.com", "123456")

    def test_send_otp_timeout_configured(self):
        """Test SMTP timeout is set to 30 seconds."""
        with patch.dict(
            "os.environ",
            {
                "EMAIL_SMTP_HOST": "smtp.test.com",
                "EMAIL_SMTP_PORT": "587",
                "EMAIL_USERNAME": "test@test.com",
                "EMAIL_PASSWORD": "testpass",
                "EMAIL_FROM_ADDRESS": "test@test.com",
            },
        ):
            with patch("agents.email_agent.smtplib.SMTP") as mock_smtp_class:
                mock_instance = MagicMock()
                mock_smtp_class.return_value.__enter__.return_value = mock_instance

                agent = EmailAgent()
                agent.send_otp("test@example.com", "123456")

                # Verify SMTP was called with timeout=30
                mock_smtp_class.assert_called_once_with(
                    "smtp.test.com", 587, timeout=30
                )
