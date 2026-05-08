import os
import asyncio
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()


class EmailAgent:
    def __init__(
        self,
        smtp_host: str | None = None,
        smtp_port: int | None = None,
        username: str | None = None,
        password: str | None = None,
        from_address: str | None = None,
        use_tls: bool = True,
    ):
        self.smtp_host = smtp_host or os.getenv("EMAIL_SMTP_HOST")
        self.smtp_port = smtp_port or int(os.getenv("EMAIL_SMTP_PORT", "587"))
        self.username = username or os.getenv("EMAIL_USERNAME")
        self.password = password or os.getenv("EMAIL_PASSWORD")
        self.from_address = from_address or os.getenv("EMAIL_FROM_ADDRESS")
        self.use_tls = use_tls

        if not self.smtp_host or not self.smtp_port or not self.username or not self.password or not self.from_address:
            raise ValueError("Email SMTP settings must be configured via environment variables")

    def send_otp(self, to_email: str, code: str, txn_details: dict | None = None) -> None:
        message = EmailMessage()
        subject = "Your OTP Code for Transaction Verification"
        body = (
            f"<html><body>"
            f"<p>Your OTP is <strong>{code}</strong>.</p>"
            f"<p>This code expires in 5 minutes.</p>"
        )
        if txn_details:
            amount = txn_details.get("amount")
            merchant = txn_details.get("merchant_id") or txn_details.get("merchant_type")
            if amount is not None or merchant is not None:
                body += "<p>Transaction details:</p><ul>"
                if amount is not None:
                    body += f"<li>Amount: {amount}</li>"
                if merchant is not None:
                    body += f"<li>Merchant: {merchant}</li>"
                body += "</ul>"
        body += "</body></html>"

        message["Subject"] = subject
        message["From"] = self.from_address
        message["To"] = to_email
        message.set_content("Your OTP code is {code}.")
        message.add_alternative(body, subtype="html")

        with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30) as smtp:
            if self.use_tls:
                smtp.starttls()
            smtp.login(self.username, self.password)
            smtp.send_message(message)

    async def send_otp_async(self, to_email: str, code: str, txn_details: dict | None = None) -> None:
        return await asyncio.to_thread(self.send_otp, to_email, code, txn_details)
