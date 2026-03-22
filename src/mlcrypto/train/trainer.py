from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from mlcrypto.data.dataset import CryptoDataset, infer_input_dim
from mlcrypto.models.factory import build_model
from mlcrypto.train.metrics import classification_metrics


@dataclass
class TrainingArtifacts:
    metrics: dict
    history: list[dict]
    checkpoint_path: Path


def _run_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_examples = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
    return total_loss / max(total_examples, 1)


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    labels = []
    probabilities = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        probs = torch.sigmoid(logits)
        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        labels.append(y.cpu().numpy())
        probabilities.append(probs.cpu().numpy())

    labels_np = np.concatenate(labels).reshape(-1)
    probabilities_np = np.concatenate(probabilities).reshape(-1)
    metrics = classification_metrics(labels_np, probabilities_np)
    metrics["loss"] = total_loss / max(total_examples, 1)
    return metrics


def train_model(
    train_path: str,
    val_path: str,
    test_path: str,
    representation: str,
    model_name: str,
    training_config: dict,
    output_dir: str | Path,
    device: str = "cpu",
) -> TrainingArtifacts:
    input_dim = infer_input_dim(train_path, representation)
    model = build_model(model_name, input_dim).to(device)

    train_ds = CryptoDataset(train_path, representation)
    val_ds = CryptoDataset(val_path, representation)
    test_ds = CryptoDataset(test_path, representation)

    batch_size = int(training_config["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    criterion = nn.BCEWithLogitsLoss()
    epochs = int(training_config["epochs"])
    patience = int(training_config["early_stopping_patience"])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"

    history = []
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = _evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_roc_auc": val_metrics["roc_auc"],
            }
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        best_state = {key: value.cpu() for key, value in model.state_dict().items()}

    torch.save(best_state, checkpoint_path)
    model.load_state_dict(best_state)
    model.to(device)
    test_metrics = _evaluate(model, test_loader, criterion, device)

    return TrainingArtifacts(metrics=test_metrics, history=history, checkpoint_path=checkpoint_path)
