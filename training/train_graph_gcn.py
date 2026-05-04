from pathlib import Path
import json
import mlflow
import sys
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

sys.path.append(str(Path(__file__).parent.parent))

from models.graph_gcn import FraudGCN

DATA_PATH = Path("data/processed/graph_data.pt")
MODEL_PATH = Path("artifacts/models/fraud_gcn_state_dict.pt")
CONFIG_PATH = Path("artifacts/models/fraud_gcn_config.json")

INPUT_SIZE = 16
HIDDEN_SIZE = 64
EMBEDDING_SIZE = 32
NUM_CLASSES = 2
DROPOUT = 0.3
LEARNING_RATE = 0.001
EPOCHS = 100

def create_masks(num_nodes: int):
    indices = torch.randperm(num_nodes)

    train_end = int(num_nodes * 0.70)
    val_end = int(num_nodes * 0.85)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def evaluate(model, data, mask):
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        predictions = torch.argmax(logits, dim=1)

    y_true = data.y[mask].detach().cpu().numpy()
    y_pred = predictions[mask].detach().cpu().numpy()
    y_score = probabilities[mask].detach().cpu().numpy()

    precision = precision_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    if len(set(y_true)) < 2:
        auc = 0.0
    else:
        auc = roc_auc_score(y_true, y_score)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
    }


def train_graph_gcn() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = torch.load(DATA_PATH, weights_only=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    train_mask, val_mask, test_mask = create_masks(data.num_nodes)

    data.train_mask = train_mask.to(device)
    data.val_mask = val_mask.to(device)
    data.test_mask = test_mask.to(device)

    model = FraudGCN(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        embedding_size=EMBEDDING_SIZE,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("fraud_anomaly_detection")

    with mlflow.start_run(run_name="graph_gcn"):
        params = {
            "model_type": "FraudGCN",
            "input_size": INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "embedding_size": EMBEDDING_SIZE,
            "num_classes": NUM_CLASSES,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
        }

        mlflow.log_params(params)

        for epoch in range(EPOCHS):
            model.train()

            optimizer.zero_grad()

            logits = model(data.x, data.edge_index)

            loss = F.cross_entropy(
                logits[data.train_mask],
                data.y[data.train_mask],
            )

            loss.backward()
            optimizer.step()

            train_metrics = evaluate(model, data, data.train_mask)
            val_metrics = evaluate(model, data, data.val_mask)

            mlflow.log_metric("train_loss", float(loss.item()), step=epoch + 1)
            mlflow.log_metric("train_f1", train_metrics["f1"], step=epoch + 1)
            mlflow.log_metric("val_f1", val_metrics["f1"], step=epoch + 1)
            mlflow.log_metric("val_auc", val_metrics["auc"], step=epoch + 1)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} "
                    f"Loss: {loss.item():.4f} "
                    f"Train F1: {train_metrics['f1']:.4f} "
                    f"Val F1: {val_metrics['f1']:.4f} "
                    f"Val AUC: {val_metrics['auc']:.4f}"
                )

        test_metrics = evaluate(model, data, data.test_mask)

        mlflow.log_metrics({
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_auc": test_metrics["auc"],
        })

        torch.save(model.state_dict(), MODEL_PATH)

        config = {
            "model_type": "FraudGCN",
            "input_size": INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "embedding_size": EMBEDDING_SIZE,
            "num_classes": NUM_CLASSES,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
        }

        with open(CONFIG_PATH, "w") as file:
            json.dump(config, file, indent=4)

        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(CONFIG_PATH))

        print()
        print("Test metrics:")
        for metric_name, metric_value in test_metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

        print()
        print(f"Saved model to: {MODEL_PATH}")
        print(f"Saved config to: {CONFIG_PATH}")


if __name__ == "__main__":
    train_graph_gcn()