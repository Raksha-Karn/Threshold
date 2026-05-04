import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv


class FraudGCN(nn.Module):
    def __init__(
        self,
        input_size: int = 16,
        hidden_size: int = 64,
        embedding_size: int = 32,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.conv1 = GraphConv(input_size, hidden_size)
        self.conv2 = GraphConv(hidden_size, embedding_size)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        logits = self.classifier(x)

        return logits