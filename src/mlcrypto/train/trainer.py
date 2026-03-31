from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from mlcrypto.data.dataset import CryptoDataset, infer_input_dim
from mlcrypto.models.factory import build_model
from mlcrypto.train.metrics import classification_metrics


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


def _is_better_checkpoint(candidate_metrics: dict, best_metrics: dict | None) -> bool:
    if best_metrics is None:
        return True

    candidate_key = (
        float(candidate_metrics["roc_auc"]),
        float(candidate_metrics["balanced_accuracy"]),
        -float(candidate_metrics["loss"]),
    )
    best_key = (
        float(best_metrics["roc_auc"]),
        float(best_metrics["balanced_accuracy"]),
        -float(best_metrics["loss"]),
    )
    return candidate_key > best_key


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
    device: str = "cpu",
    train_seed: int | None = None,
) -> dict:
    input_dim = infer_input_dim(train_path, representation)
    model = build_model(model_name, input_dim).to(device)

    train_ds = CryptoDataset(train_path, representation)
    val_ds = CryptoDataset(val_path, representation)
    test_ds = CryptoDataset(test_path, representation)

    batch_size = int(training_config["batch_size"])
    loader_generator = None
    if train_seed is not None:
        loader_generator = torch.Generator()
        loader_generator.manual_seed(int(train_seed))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=loader_generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training_config.get("scheduler_factor", 0.5)),
        patience=int(training_config.get("scheduler_patience", 2)),
    )
    criterion = nn.BCEWithLogitsLoss()
    epochs = int(training_config["epochs"])
    patience = int(training_config["early_stopping_patience"])

    best_state = None
    best_metrics = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        _run_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = _evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])

        if _is_better_checkpoint(val_metrics, best_metrics):
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            best_metrics = dict(val_metrics)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        best_state = {key: value.cpu() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(device)
    test_metrics = _evaluate(model, test_loader, criterion, device)

    return test_metrics
