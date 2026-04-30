import random
import uuid
import os
from datetime import datetime, timedelta
from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

NUM_USERS = 500
NUM_TX = 5000
NUM_RINGS = 50

def create_users(tx):
    users = []
    for i in range(NUM_USERS):
        user_id = str(uuid.uuid4())
        users.append(user_id)
        tx.run(
            """
            MERGE (u:User {id: $id})
            SET u.email = $email,
                u.createdAt = $created_at
            """,
            id=user_id,
            email=f"user{i}@test.com",
            created_at = datetime.utcnow().isoformat()
        )
    return users

def create_transactions(tx, users):
    for _ in range(NUM_TX):
        sender = random.choice(users)
        receiver = random.choice(users)
        if sender == receiver:
            continue
        tx_id = str(uuid.uuid4())

        tx.run("""
                MERGE (s:User {id:$sender})
                MERGE (r:User {id:$receiver})
                MERGE (t:Transaction {id:$tx_id})
                SET t.amount = $amount,
                    t.timestamp = $timestamp
               
                MERGE (s)-[:SENT]->(t)
                MERGE (t)-[:TO]->(r)
               """,
               sender=sender,
               receiver=receiver,
               tx_id=tx_id,
               amount = round(random.uniform(5, 5000), 2),
               timestamp = (datetime.utcnow() - timedelta(days=random.randint(0, 30))).isoformat()
            )
        
def create_fraud_rings(tx, users):
    for _ in range(NUM_RINGS):
        ring_users = random.sample(users, 5)
        device_id = str(uuid.uuid4())
        for user in ring_users:
            tx.run(
                """
                MERGE (u:User {id:$user_id})
                MERGE (d:Device {id:$device_id})
                MERGE (u)-[:LINKED_TO]->(d)
                """,
                user_id=user,
                device_id=device_id
            )

        # circular transaction case
        for i in range(len(ring_users)):
            sender = ring_users[i]
            receiver = ring_users[(i+1) % len(ring_users)]
            tx_id = str(uuid.uuid4())

            tx.run(
                """
                MERGE (s:User {id:$sender})
                MERGE (r:User {id:$receiver})
                MERGE (t:Transaction {id:$tx_id})
                SET t.timestamp = $timestamp,
                    t.flagged = true,
                    t.amount = $amount
                MERGE (s)-[:SENT]->(t)
                MERGE (t)-[:TO]->(r)
                """,
                sender=sender,
                receiver=receiver,
                tx_id=tx_id, 
                amount=round(random.uniform(1000, 10000), 2), 
                timestamp=datetime.utcnow().isoformat()
            )

def main():
    with driver.session() as session:
        print("Creating Users...")
        users = session.execute_write(create_users)

        print("Creating transactions...")
        transactions = session.execute_write(create_transactions, users)

        print("Creating fraud rings...")
        fraud_rings = session.execute_write(create_fraud_rings, users)

    driver.close()
    print("Seeding done")

if __name__ == "__main__":
    main()