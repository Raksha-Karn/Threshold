from prometheus_client import Counter, Histogram

REQUEST_LATENCY = Histogram(
    "txn_processing_latency_ms",
    "Transaction processing latency in milliseconds",
    buckets=[50, 100, 200, 400, 800, 1200, 2000],
)
FRAUD_SCORE_HISTOGRAM = Histogram(
    "fraud_score_histogram",
    "Fraud score histogram",
    buckets=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
)
VERDICT_COUNTER = Counter(
    "txn_verdict_total",
    "Count of transaction verdicts",
    ["verdict"],
)
OTP_SUCCESS_COUNTER = Counter(
    "otp_success_total",
    "Count of successful OTP verifications",
)
OTP_FAILURE_COUNTER = Counter(
    "otp_failure_total",
    "Count of failed OTP verifications",
)
