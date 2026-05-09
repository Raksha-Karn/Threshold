from dataclasses import dataclass


GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


@dataclass
class DemoTxn:
    txn_id: str
    scenario: str
    amount: float
    risk_score: float
    expected: str


def verdict_for_score(score: float) -> str:
    if score < 0.3:
        return "APPROVED"
    if score < 0.7:
        return "OTP_REQUIRED"
    return "BLOCKED"


def color(verdict: str) -> str:
    if verdict == "APPROVED":
        return f"{GREEN}{verdict}{RESET}"
    if verdict == "OTP_REQUIRED":
        return f"{YELLOW}{verdict}{RESET}"
    return f"{RED}{verdict}{RESET}"


def demo_transactions() -> list[DemoTxn]:
    return [
        DemoTxn("demo_001", "normal grocery", 42.50, 0.12, "APPROVED"),
        DemoTxn("demo_002", "2am flight ticket", 1200.00, 0.42, "OTP_REQUIRED"),
        DemoTxn("demo_003", "cold start transfer", 5000.00, 0.65, "OTP_REQUIRED"),
        DemoTxn("demo_004", "known high-risk merchant", 850.00, 0.72, "BLOCKED"),
        DemoTxn("demo_005", "SIM swap OTP", 750.00, 0.91, "BLOCKED"),
        DemoTxn("demo_006", "fraud ring shared device", 990.00, 0.88, "BLOCKED"),
        DemoTxn("demo_007", "impossible travel", 300.00, 0.76, "BLOCKED"),
        DemoTxn("demo_008", "velocity burst", 99.00, 0.81, "BLOCKED"),
        DemoTxn("demo_009", "trusted merchant", 65.00, 0.18, "APPROVED"),
        DemoTxn("demo_010", "medium-risk ecommerce", 410.00, 0.51, "OTP_REQUIRED"),
    ]


def main():
    rows = []
    for txn in demo_transactions():
        actual = verdict_for_score(txn.risk_score)
        passed = "PASS" if actual == txn.expected else "FAIL"
        rows.append((txn, actual, passed))

    print("Fraud Detection Demo")
    print("-" * 92)
    print(f"{'TXN ID':<12} {'SCENARIO':<28} {'AMOUNT':>10} {'SCORE':>8} {'EXPECTED':<14} {'ACTUAL':<22} RESULT")
    print("-" * 92)
    for txn, actual, passed in rows:
        result_color = GREEN if passed == "PASS" else RED
        print(
            f"{txn.txn_id:<12} "
            f"{txn.scenario:<28} "
            f"${txn.amount:>9.2f} "
            f"{txn.risk_score:>8.2f} "
            f"{txn.expected:<14} "
            f"{color(actual):<22} "
            f"{result_color}{passed}{RESET}"
        )
    print("-" * 92)


if __name__ == "__main__":
    main()
