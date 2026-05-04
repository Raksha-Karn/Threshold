import random
import uuid
import os
from datetime import datetime, timedelta

from neo4j import GraphDatabase


NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD),
)

NUM_USERS = 500
NUM_TX = 5000
NUM_RINGS = 50
RING_SIZE = 5


def clear_database(tx):
    tx.run("MATCH (n) DETACH DELETE n")


def create_constraints(tx):
    tx.run("CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT tx_id_unique IF NOT EXISTS FOR (t:Transaction) REQUIRE t.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT device_id_unique IF NOT EXISTS FOR (d:Device) REQUIRE d.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT ip_id_unique IF NOT EXISTS FOR (ip:IP) REQUIRE ip.id IS UNIQUE")


def create_users(tx):
    users = []

    for i in range(NUM_USERS):
        user_id = str(uuid.uuid4())
        users.append(user_id)

        tx.run(
            """
            MERGE (u:User {id: $id})
            SET u.email = $email,
                u.createdAt = $created_at,
                u.is_fraud = 0,
                u.linked_device_count = 1,
                u.shared_ip_count = 0,
                u.ring_density = 0.0,
                u.avg_geo_distance_from_home = $avg_geo_distance_from_home,
                u.max_geo_distance_from_home = $max_geo_distance_from_home,
                u.is_international_rate = $is_international_rate,
                u.txn_count_last_24h = $txn_count_last_24h,
                u.amount_sum_last_24h = $amount_sum_last_24h,
                u.fraud_score_mean = 0.0,
                u.anomaly_score_mean = $anomaly_score_mean,
                u.unique_merchants = $unique_merchants,
                u.unique_devices = 1,
                u.new_device_rate = 0.0,
                u.new_city_rate = $new_city_rate,
                u.card_present_rate = $card_present_rate,
                u.merchant_risk_mean = $merchant_risk_mean
            """,
            id=user_id,
            email=f"user{i}@test.com",
            created_at=datetime.utcnow().isoformat(),
            avg_geo_distance_from_home=random.uniform(1, 80),
            max_geo_distance_from_home=random.uniform(20, 300),
            is_international_rate=random.uniform(0, 0.1),
            txn_count_last_24h=random.randint(0, 5),
            amount_sum_last_24h=random.uniform(0, 2000),
            anomaly_score_mean=random.uniform(0, 0.3),
            unique_merchants=random.randint(1, 8),
            new_city_rate=random.uniform(0, 0.2),
            card_present_rate=random.uniform(0.5, 1.0),
            merchant_risk_mean=random.uniform(0, 0.2),
        )

    return users


def create_normal_devices_and_ips(tx, users):
    for user in users:
        device_id = f"device_{uuid.uuid4()}"
        ip_id = f"ip_{uuid.uuid4()}"

        tx.run(
            """
            MATCH (u:User {id: $user_id})
            MERGE (d:Device {id: $device_id})
            MERGE (ip:IP {id: $ip_id})
            MERGE (u)-[:USES_DEVICE]->(d)
            MERGE (u)-[:USES_IP]->(ip)
            """,
            user_id=user,
            device_id=device_id,
            ip_id=ip_id,
        )


def create_transactions(tx, users):
    for _ in range(NUM_TX):
        sender = random.choice(users)
        receiver = random.choice(users)

        if sender == receiver:
            continue

        tx_id = str(uuid.uuid4())

        tx.run(
            """
            MERGE (s:User {id:$sender})
            MERGE (r:User {id:$receiver})
            MERGE (t:Transaction {id:$tx_id})
            SET t.amount = $amount,
                t.timestamp = $timestamp,
                t.flagged = false

            MERGE (s)-[:SENT]->(t)
            MERGE (t)-[:TO]->(r)
            MERGE (s)-[:TRANSACTED_WITH]->(r)
            MERGE (r)-[:TRANSACTED_WITH]->(s)
            """,
            sender=sender,
            receiver=receiver,
            tx_id=tx_id,
            amount=round(random.uniform(5, 5000), 2),
            timestamp=(datetime.utcnow() - timedelta(days=random.randint(0, 30))).isoformat(),
        )


def create_fraud_rings(tx, users):
    fraud_users = set()

    for _ in range(NUM_RINGS):
        ring_users = random.sample(users, RING_SIZE)
        fraud_users.update(ring_users)

        shared_device_id = f"shared_device_{uuid.uuid4()}"
        shared_ip_id = f"shared_ip_{uuid.uuid4()}"

        for user in ring_users:
            tx.run(
                """
                MATCH (u:User {id:$user_id})
                MERGE (d:Device {id:$device_id})
                MERGE (ip:IP {id:$ip_id})
                MERGE (u)-[:USES_DEVICE]->(d)
                MERGE (u)-[:USES_IP]->(ip)
                SET u.is_fraud = 1,
                    u.linked_device_count = 5,
                    u.shared_ip_count = 5,
                    u.ring_density = 1.0,
                    u.avg_geo_distance_from_home = $avg_geo_distance_from_home,
                    u.max_geo_distance_from_home = $max_geo_distance_from_home,
                    u.is_international_rate = $is_international_rate,
                    u.txn_count_last_24h = $txn_count_last_24h,
                    u.amount_sum_last_24h = $amount_sum_last_24h,
                    u.fraud_score_mean = $fraud_score_mean,
                    u.anomaly_score_mean = $anomaly_score_mean,
                    u.unique_merchants = $unique_merchants,
                    u.unique_devices = 5,
                    u.new_device_rate = $new_device_rate,
                    u.new_city_rate = $new_city_rate,
                    u.card_present_rate = $card_present_rate,
                    u.merchant_risk_mean = $merchant_risk_mean
                """,
                user_id=user,
                device_id=shared_device_id,
                ip_id=shared_ip_id,
                avg_geo_distance_from_home=random.uniform(300, 2500),
                max_geo_distance_from_home=random.uniform(1000, 8000),
                is_international_rate=random.uniform(0.3, 1.0),
                txn_count_last_24h=random.randint(5, 30),
                amount_sum_last_24h=random.uniform(3000, 50000),
                fraud_score_mean=random.uniform(0.6, 1.0),
                anomaly_score_mean=random.uniform(0.5, 1.0),
                unique_merchants=random.randint(5, 30),
                new_device_rate=random.uniform(0.4, 1.0),
                new_city_rate=random.uniform(0.3, 1.0),
                card_present_rate=random.uniform(0.0, 0.5),
                merchant_risk_mean=random.uniform(0.5, 1.0),
            )

        for i in range(len(ring_users)):
            for j in range(i + 1, len(ring_users)):
                tx.run(
                    """
                    MATCH (u1:User {id:$u1})
                    MATCH (u2:User {id:$u2})
                    MERGE (u1)-[:LINKED_TO]->(u2)
                    MERGE (u2)-[:LINKED_TO]->(u1)
                    """,
                    u1=ring_users[i],
                    u2=ring_users[j],
                )

        for i in range(len(ring_users)):
            sender = ring_users[i]
            receiver = ring_users[(i + 1) % len(ring_users)]
            tx_id = str(uuid.uuid4())

            tx.run(
                """
                MATCH (s:User {id:$sender})
                MATCH (r:User {id:$receiver})
                MERGE (t:Transaction {id:$tx_id})
                SET t.timestamp = $timestamp,
                    t.flagged = true,
                    t.amount = $amount

                MERGE (s)-[:SENT]->(t)
                MERGE (t)-[:TO]->(r)
                MERGE (s)-[:TRANSACTED_WITH]->(r)
                MERGE (r)-[:TRANSACTED_WITH]->(s)
                """,
                sender=sender,
                receiver=receiver,
                tx_id=tx_id,
                amount=round(random.uniform(1000, 10000), 2),
                timestamp=datetime.utcnow().isoformat(),
            )

    return list(fraud_users)


def main():
    with driver.session() as session:
        print("Clearing database...")
        session.execute_write(clear_database)

        print("Creating constraints...")
        session.execute_write(create_constraints)

        print("Creating users...")
        users = session.execute_write(create_users)

        print("Creating normal devices and IPs...")
        session.execute_write(create_normal_devices_and_ips, users)

        print("Creating transactions...")
        session.execute_write(create_transactions, users)

        print("Creating fraud rings...")
        fraud_users = session.execute_write(create_fraud_rings, users)

    driver.close()

    print("Seeding done")
    print(f"Users: {NUM_USERS}")
    print(f"Transactions: about {NUM_TX + NUM_RINGS * RING_SIZE}")
    print(f"Fraud users: {len(set(fraud_users))}")


if __name__ == "__main__":
    main()