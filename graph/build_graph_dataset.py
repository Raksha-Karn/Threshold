from pathlib import Path
import json
import os
import torch
from torch_geometric.data import Data
from neo4j import GraphDatabase


NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

OUTPUT_PATH = Path("data/processed/graph_data.pt")
METADATA_PATH = Path("data/processed/graph_metadata.json")


NODE_FEATURES = [
    "linked_device_count",
    "shared_ip_count",
    "ring_density",
    "avg_geo_distance_from_home",
    "max_geo_distance_from_home",
    "is_international_rate",
    "txn_count_last_24h",
    "amount_sum_last_24h",
    "fraud_score_mean",
    "anomaly_score_mean",
    "unique_merchants",
    "unique_devices",
    "new_device_rate",
    "new_city_rate",
    "card_present_rate",
    "merchant_risk_mean",
]


def fetch_user_nodes(driver):
    query = """
    MATCH (u:User)
    RETURN
        u.id AS user_id,
        coalesce(u.linked_device_count, 0.0) AS linked_device_count,
        coalesce(u.shared_ip_count, 0.0) AS shared_ip_count,
        coalesce(u.ring_density, 0.0) AS ring_density,
        coalesce(u.avg_geo_distance_from_home, 0.0) AS avg_geo_distance_from_home,
        coalesce(u.max_geo_distance_from_home, 0.0) AS max_geo_distance_from_home,
        coalesce(u.is_international_rate, 0.0) AS is_international_rate,
        coalesce(u.txn_count_last_24h, 0.0) AS txn_count_last_24h,
        coalesce(u.amount_sum_last_24h, 0.0) AS amount_sum_last_24h,
        coalesce(u.fraud_score_mean, 0.0) AS fraud_score_mean,
        coalesce(u.anomaly_score_mean, 0.0) AS anomaly_score_mean,
        coalesce(u.unique_merchants, 0.0) AS unique_merchants,
        coalesce(u.unique_devices, 0.0) AS unique_devices,
        coalesce(u.new_device_rate, 0.0) AS new_device_rate,
        coalesce(u.new_city_rate, 0.0) AS new_city_rate,
        coalesce(u.card_present_rate, 0.0) AS card_present_rate,
        coalesce(u.merchant_risk_mean, 0.0) AS merchant_risk_mean,
        coalesce(u.is_fraud, 0) AS label
    ORDER BY user_id
    """

    records, _, _ = driver.execute_query(query)
    return [record.data() for record in records]


def fetch_user_edges(driver):
    query = """
    MATCH (u1:User)-[:LINKED_TO]-(u2:User)
    RETURN DISTINCT u1.id AS source_user_id, u2.id AS target_user_id
    """

    records, _, _ = driver.execute_query(query)
    return [record.data() for record in records]


def build_graph_dataset() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )

    try:
        node_rows = fetch_user_nodes(driver)
        edge_rows = fetch_user_edges(driver)
    finally:
        driver.close()

    if not node_rows:
        raise ValueError("No User nodes found in Neo4j.")

    user_ids = [row["user_id"] for row in node_rows]

    user_id_to_index = {
        user_id: index
        for index, user_id in enumerate(user_ids)
    }

    x = torch.tensor(
        [
            [float(row[feature]) for feature in NODE_FEATURES]
            for row in node_rows
        ],
        dtype=torch.float32,
    )

    y = torch.tensor(
        [int(row["label"]) for row in node_rows],
        dtype=torch.long,
    )

    edges = []

    for row in edge_rows:
        source_user_id = row["source_user_id"]
        target_user_id = row["target_user_id"]

        if source_user_id not in user_id_to_index:
            continue

        if target_user_id not in user_id_to_index:
            continue

        source_index = user_id_to_index[source_user_id]
        target_index = user_id_to_index[target_user_id]

        edges.append([source_index, target_index])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
    )

    torch.save(data, OUTPUT_PATH)

    metadata = {
        "node_features": NODE_FEATURES,
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.shape[1]),
        "num_fraud_nodes": int((data.y == 1).sum().item()),
        "num_normal_nodes": int((data.y == 0).sum().item()),
        "user_ids": user_ids,
        "user_id_to_index": user_id_to_index,
    }

    with open(METADATA_PATH, "w") as file:
        json.dump(metadata, file, indent=4)

    print(f"Saved graph data to: {OUTPUT_PATH}")
    print(f"Saved graph metadata to: {METADATA_PATH}")
    print(f"Nodes: {data.num_nodes}")
    print(f"Edges: {data.edge_index.shape[1]}")
    print(f"Feature shape: {data.x.shape}")
    print(f"Fraud nodes: {metadata['num_fraud_nodes']}")
    print(f"Normal nodes: {metadata['num_normal_nodes']}")


if __name__ == "__main__":
    build_graph_dataset()