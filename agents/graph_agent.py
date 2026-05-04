import json

import torch

from models.graph_gcn import FraudGCN


class GraphAgent:
    def __init__(
        self,
        model_path: str = "artifacts/models/fraud_gcn_state_dict.pt",
        config_path: str = "artifacts/models/fraud_gcn_config.json",
        metadata_path: str = "data/processed/graph_metadata.json",
        graph_data_path: str = "data/processed/graph_data.pt",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(config_path, "r") as file:
            self.config = json.load(file)

        with open(metadata_path, "r") as file:
            self.metadata = json.load(file)

        self.user_id_to_index = self.metadata["user_id_to_index"]

        self.data = torch.load(
            graph_data_path,
            map_location=self.device,
            weights_only=False,
        ).to(self.device)

        self.model = FraudGCN(
            input_size=self.config["input_size"],
            hidden_size=self.config["hidden_size"],
            embedding_size=self.config["embedding_size"],
            num_classes=self.config["num_classes"],
            dropout=self.config["dropout"],
        ).to(self.device)

        state_dict = torch.load(
            model_path,
            map_location=self.device,
        )

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def score(self, user_id: str) -> float:
        if user_id not in self.user_id_to_index:
            return 0.0

        node_index = self.user_id_to_index[user_id]

        with torch.no_grad():
            logits = self.model(self.data.x, self.data.edge_index)
            probabilities = torch.softmax(logits, dim=1)
            fraud_probability = probabilities[node_index, 1].item()

        return float(fraud_probability)

    def is_suspicious(self, user_id: str, threshold: float = 0.5) -> bool:
        return self.score(user_id) >= threshold