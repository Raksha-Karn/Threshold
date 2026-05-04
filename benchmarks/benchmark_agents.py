import json
import statistics
import time
from pathlib import Path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.anomaly_agent import AnomalyAgent
from agents.velocity_rules_agent import VelocityRulesAgent
from agents.graph_agent import GraphAgent
from agents.geo_agent import GeoAgent


BENCHMARK_ITERATIONS = 200
LATENCY_TARGET_MS = 100.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0

    values = sorted(values)
    index = int(len(values) * p / 100)
    index = min(index, len(values) - 1)

    return values[index]


def benchmark_agent(name: str, fn, iterations: int = BENCHMARK_ITERATIONS) -> dict:
    latencies_ms = []

    fn()

    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies_ms.append(latency_ms)

    avg_ms = statistics.mean(latencies_ms)
    p95_ms = percentile(latencies_ms, 95)
    min_ms = min(latencies_ms)
    max_ms = max(latencies_ms)

    print(
        f"{name}: "
        f"avg={avg_ms:.2f}ms "
        f"p95={p95_ms:.2f}ms "
        f"min={min_ms:.2f}ms "
        f"max={max_ms:.2f}ms"
    )

    return {
        "name": name,
        "avg_ms": avg_ms,
        "p95_ms": p95_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
    }


def load_benchmark_user_id() -> str:
    metadata_path = Path("data/processed/graph_metadata.json")

    if not metadata_path.exists():
        return "benchmark_user"

    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    user_ids = metadata.get("user_ids", [])

    if not user_ids:
        return "benchmark_user"

    return user_ids[0]


def preload_velocity_history(
    velocity_agent: VelocityRulesAgent,
    user_id: str,
    base_time: float,
) -> None:
    redis_key = velocity_agent._user_key(user_id)

    velocity_agent.redis_client.delete(redis_key)

    for i in range(10):
        historical_transaction = {
            "user_id": user_id,
            "timestamp_epoch": base_time - ((i + 1) * 300),  # every 5 minutes
            "amount": 100.0 + i,
            "merchant_id": f"merchant_{i % 4}",
            "device_id": "device_benchmark",
            "city": "Kathmandu",
            "geo_distance_from_home": 10.0,
            "card_present_flag": 1,
            "is_international": 0,
        }

        velocity_agent.score(
            historical_transaction,
            update_redis=True,
        )


def main() -> None:
    print("Loading agents...")

    anomaly_agent = AnomalyAgent()
    velocity_agent = VelocityRulesAgent()
    graph_agent = GraphAgent()
    geo_agent = GeoAgent()

    user_id = load_benchmark_user_id()
    base_time = time.time()

    benchmark_transaction = {
        "user_id": user_id,

        "timestamp_epoch": base_time,
        "timestamp": "2026-05-04T10:00:00+00:00",
        "amount": 2500.0,
        "merchant_id": "merchant_benchmark",
        "device_id": "device_benchmark",
        "city": "Kathmandu",

        "lat": 40.7128,
        "lon": -74.0060,
        "home_lat": 27.7172,
        "home_lon": 85.3240,
        "geo_distance_from_home": 1500.0,
        "card_present_flag": 0,
        "is_international": 1,

        "hour_of_day": 2,
        "day_of_week": 5,
        "amount_zscore": 4.0,
        "user_mean_amount": 200.0,
        "amount_vs_user_mean": 12.5,
        "txn_count_in_last_1h": 5,
        "txn_count_in_last_24h": 20,
        "is_new_merchant": 1,
        "anomaly_score": 0.95,
        "fraud_score": 0.95,
    }

    print("Preparing Redis benchmark history...")
    preload_velocity_history(
        velocity_agent=velocity_agent,
        user_id=user_id,
        base_time=base_time,
    )

    print()
    print(f"Running benchmark with {BENCHMARK_ITERATIONS} iterations...")
    print()

    results = [
        benchmark_agent(
            "AnomalyAgent",
            lambda: anomaly_agent.score(benchmark_transaction),
        ),
        benchmark_agent(
            "VelocityRulesAgent",
            lambda: velocity_agent.score(
                benchmark_transaction,
                update_redis=False,
            ),
        ),
        benchmark_agent(
            "GraphAgent",
            lambda: graph_agent.score(user_id),
        ),
        benchmark_agent(
            "GeoAgent",
            lambda: geo_agent.score(
                benchmark_transaction,
                update_history=False,
            ),
        ),
    ]

    print()
    print(f"Latency target: p95 < {LATENCY_TARGET_MS:.0f}ms")

    for result in results:
        status = "PASS" if result["p95_ms"] < LATENCY_TARGET_MS else "FAIL"
        print(f"{result['name']}: {status}")

    print()
    print("Benchmark summary:")
    for result in results:
        print(
            f"{result['name']}: "
            f"avg={result['avg_ms']:.2f}ms, "
            f"p95={result['p95_ms']:.2f}ms"
        )


if __name__ == "__main__":
    main()