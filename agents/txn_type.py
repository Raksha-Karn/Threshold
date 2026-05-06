from enum import Enum


class TransactionType(Enum):
    HIGH_VALUE_P2P = "HIGH_VALUE_P2P"
    POS_RETAIL = "POS_RETAIL"
    QR_CODE = "QR_CODE"
    REMITTANCE = "REMITTANCE"
    ATM = "ATM"
    ONLINE_MERCHANT = "ONLINE_MERCHANT"