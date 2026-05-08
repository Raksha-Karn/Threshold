import os
import asyncio
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()


class SMSAgent:
    def __init__(
        self,
        account_sid: str | None = None,
        auth_token: str | None = None,
        from_number: str | None = None,
    ):
        self.account_sid = account_sid or os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = auth_token or os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = from_number or os.getenv("TWILIO_FROM_NUMBER")

        if not self.account_sid or not self.auth_token or not self.from_number:
            raise ValueError("Twilio credentials and from number must be set in environment variables")

        self.client = Client(self.account_sid, self.auth_token)

    def send_otp(self, to_phone: str, code: str, txn_details: dict | None = None) -> str:
        body = f"Your transaction OTP is {code}. It expires in 5 minutes."
        if txn_details:
            merchant = txn_details.get("merchant_id") or txn_details.get("merchant_type")
            amount = txn_details.get("amount")
            if merchant or amount is not None:
                body += "\n"
                if amount is not None:
                    body += f"Amount: {amount}. "
                if merchant is not None:
                    body += f"Merchant: {merchant}."

        message = self.client.messages.create(
            body=body,
            from_=self.from_number,
            to=to_phone,
        )
        return message.sid

    async def send_otp_async(self, to_phone: str, code: str, txn_details: dict | None = None) -> str:
        return await asyncio.to_thread(self.send_otp, to_phone, code, txn_details)
