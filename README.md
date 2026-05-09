# Threshold

Agentic fraud detection framework with a FastAPI orchestrator, Redis velocity state, Neo4j graph context, Kafka real-time ingestion, MLflow model tracking, OTP interlock, Prometheus metrics, and scenario tests.

## Architecture

```text
producer -> Kafka txn.raw -> KafkaFraudConsumer -> FastAPI/FraudOrchestrator
                                               -> validate input
                                               -> Redis velocity rules
                                               -> SynthesisAgent concurrent model scoring
                                               -> threshold verdict
                                               -> OTPInterlock when needed
                                               -> Redis status + MLflow prediction + Prometheus metrics
                                               -> Kafka txn.verdict -> MockPaymentProcessor
                                               -> Kafka txn.dlq on processing errors

Data services:
Redis: velocity windows, transaction status, OTP state
Neo4j: user/device/SIM-swap graph context
MLflow: model registry, prediction events, daily performance
Prometheus endpoint: /metrics
Dashboard: /dashboard
```

## Setup

Install dependencies:

```bash
uv sync
```

Run the full stack:

```bash
docker compose up --build
```

Core services:

```text
FastAPI: http://localhost:8000
Dashboard: http://localhost:8000/dashboard
Metrics: http://localhost:8000/metrics
MLflow UI: http://localhost:5000
Neo4j Browser: http://localhost:7474
Kafka bootstrap: localhost:9092
Redis: localhost:6379
```

Run FastAPI locally when Redis and Neo4j are already available:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API

Process a transaction:

```bash
curl -X POST http://localhost:8000/transaction \
  -H 'Content-Type: application/json' \
  -d '{"transaction_id":"txn_1","user_id":"user_1","amount":100,"merchant_id":"merchant_1","txn_type":"POS_RETAIL","phone":"+15550000001","email":"user@example.com"}'
```

Verify OTP:

```bash
curl -X POST http://localhost:8000/otp/verify \
  -H 'Content-Type: application/json' \
  -d '{"txn_id":"txn_1","sms_code":"123456","email_code":"654321"}'
```

Read status:

```bash
curl http://localhost:8000/transaction/txn_1/status
```

## Kafka Pipeline

Topics are created with 3 partitions by `kafka-init`:

```text
txn.raw
txn.verdict
txn.dlq
```

`KafkaFraudConsumer` reads `txn.raw`, calls the orchestrator, publishes verdicts to `txn.verdict`, and publishes failed messages to `txn.dlq`. `MockPaymentProcessor` consumes `txn.verdict` and maps `APPROVED`, `BLOCKED`, and `OTP_REQUIRED` to payment actions.

## MLflow Registry And A/B Routing

`registry/model_registry.py` includes:

- `ModelVersionManager.register_all_v1()` for `isolation_forest/v1`, `lstm_behavior/v1`, `velocity_random_forest/v1`, and `graph_gcn/v1`
- environment tags and aliases for `Staging`, `Production`, and `Archived`
- `ModelRouter`, which routes 90 percent of traffic to Production and 10 percent to Staging
- daily performance logging for `fraud_catch_rate`, `false_positive_rate`, and `avg_latency`
- `DriftDetector`, which alerts and marks retraining required when fraud catch rate drops by more than 5 percent week over week

Register the baseline model versions after MLflow is running:

```bash
python -m registry.register_models
```

## Tests

Focused tests for the new work:

```bash
pytest tests/test_fraud_scenarios.py tests/test_model_registry.py tests/test_fraud_orchestrator.py tests/test_kafka_pipeline.py
```

Scenario coverage:

```text
Scenario 1: legit 2am flight ticket -> APPROVED or OTP_REQUIRED, not BLOCKED
Scenario 2: cold-start $5000 transfer -> OTP_REQUIRED or BLOCKED
Scenario 3: SIM swap + fast OTP -> BLOCKED and frozen/escalated
Scenario 4: shared-device fraud ring -> graph detector flags
Scenario 5: Kathmandu then London in 30 minutes -> geo detector flags
Scenario 6: 15 transactions in 10 minutes -> velocity detector flags
```

## Demo

Run the terminal demo:

```bash
python demo.py
```

It prints 10 mixed transactions with expected and actual verdicts.

## Retraining Pipeline

1. Land new labeled transaction data under `data/raw/`.
2. Rebuild features with `features/feature_engineering.py`, `features/build_velocity_features.py`, and `features/build_sequences.py`.
3. Retrain models:

```bash
python training/train_velocity_random_forest.py
python training/train_behaviour_lstm.py
python training/train_graph_gcn.py
python models/isolation_forest.py
```

4. Recalibrate synthesis thresholds:

```bash
python training/calibrate_synthesis.py
```

5. Register new versions in MLflow using `ModelVersionManager`.
6. Route 10 percent of traffic to Staging with `ModelRouter`.
7. Compare daily `fraud_catch_rate`, `false_positive_rate`, and `avg_latency`.
8. Promote to Production when staging metrics are better or equivalent and P95 latency remains below 800 ms.

## Current Verification Notes

The code includes the runtime hooks for the 800 ms target, P50/P95/P99 benchmark reporting, and Prometheus latency histograms. A final true end-to-end latency result requires Redis, Neo4j, Kafka, and MLflow running together, because sandboxed local network access can block service connections.
