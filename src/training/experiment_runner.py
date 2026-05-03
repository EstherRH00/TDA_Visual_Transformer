import os
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def get_class_weights(labels):
    """Compute pos_weight for BCEWithLogitsLoss to handle class imbalance.

    Args:
        - labels: list or array of binary labels (0 or 1).

    Returns:
        - pos_weight: torch tensor with the ratio of negative to positive samples.
    """
    labels = np.array(labels)
    n_neg = (labels == 0).sum()
    n_pos = (labels == 1).sum()
    return torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Run one training epoch over the dataloader.

    Handles both 2-tuple (image, label) and 3-tuple (image, tda, label) batches.

    Args:
        - model: PyTorch model to train.
        - dataloader: DataLoader yielding batches.
        - optimizer: optimizer instance.
        - criterion: loss function.
        - device: 'cuda' or 'cpu'.

    Returns:
        - avg_loss: average training loss over all batches.
    """
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        if len(batch) == 3:
            x, tda, y = batch
            x, tda, y = x.to(device), tda.to(device), y.to(device)
            out = model(x, tda)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            out = model(x)
        loss = criterion(out.squeeze(), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate the model on a dataloader, computing loss and classification metrics.

    Args:
        - model: PyTorch model to evaluate.
        - dataloader: DataLoader yielding batches.
        - criterion: loss function.
        - device: 'cuda' or 'cpu'.

    Returns:
        - metrics: dict with 'loss', 'accuracy', 'f1', and 'auc'.
        - y_true: list of ground-truth labels.
        - y_prob: list of predicted probabilities.
    """
    model.eval()
    total_loss = 0.0
    all_y, all_p = [], []
    for batch in dataloader:
        if len(batch) == 3:
            x, tda, y = batch
            x, tda, y = x.to(device), tda.to(device), y.to(device)
            out = model(x, tda)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            out = model(x)
        loss = criterion(out.squeeze(), y)
        total_loss += loss.item()
        probs = torch.sigmoid(out.squeeze()).cpu().numpy()
        all_p.extend(probs.tolist() if probs.ndim > 0 else [probs.item()])
        all_y.extend(y.cpu().numpy().tolist() if y.ndim > 0 else [y.cpu().item()])

    all_y = np.array(all_y)
    all_p = np.array(all_p)
    preds = (all_p > 0.5).astype(int)

    metrics = {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy_score(all_y, preds),
        "f1": f1_score(all_y, preds, zero_division=0),
        "auc": roc_auc_score(all_y, all_p) if len(np.unique(all_y)) > 1 else 0.0,
    }
    return metrics, all_y.tolist(), all_p.tolist()


def run_experiment(model_fn, train_dataset, test_dataset, config):
    """Orchestrate a full experiment: train/val split, training with early stopping, and test evaluation.

    Results (metrics, history, predictions) are persisted to a JSON file for survival across kernel restarts.

    Args:
        - model_fn: callable that returns a fresh model instance.
        - train_dataset: full training dataset (will be split into train/val).
        - test_dataset: held-out test dataset.
        - config: dict with keys 'experiment_name' (str), 'seed' (int, default 2),
          'epochs' (int, default 30), 'patience' (int, default 5), 'lr' (float, default 1e-4),
          'batch_size' (int, default 16), 'save_dir' (str, default 'checkpoints').

    Returns:
        - result: dict with 'test' (metrics), 'history', 'checkpoint' path, 'y_true', 'y_prob'.
    """
    seed = config.get("seed", 2)
    epochs = config.get("epochs", 30)
    patience = config.get("patience", 5)
    lr = config.get("lr", 1e-4)
    batch_size = config.get("batch_size", 16)
    save_dir = config.get("save_dir", "checkpoints")
    name = config.get("experiment_name", "experiment")

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Train/Val split (stratified 85/15) ---
    labels = [train_dataset.labels[i] for i in range(len(train_dataset))]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))

    train_sub = Subset(train_dataset, train_idx)
    val_sub = Subset(train_dataset, val_idx)

    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    # --- Model, loss, optimizer ---
    model = model_fn().to(device)
    pos_weight = get_class_weights(labels).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # --- Training loop with early stopping ---
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"{name}_best.pt")

    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_auc"].append(val_metrics["auc"])

        print(f"  [{name}] Epoch {epoch}/{epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  "
              f"val_auc={val_metrics['auc']:.4f}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  [{name}] Early stopping at epoch {epoch}")
                break

    # --- Save best & evaluate on test ---
    torch.save(best_state, ckpt_path)
    model.load_state_dict(best_state)
    test_metrics, y_true, y_prob = evaluate(model, test_loader, criterion, device)

    print(f"  [{name}] TEST  acc={test_metrics['accuracy']:.4f}  "
          f"f1={test_metrics['f1']:.4f}  auc={test_metrics['auc']:.4f}")

    # --- Save results to JSON for persistence across kernel restarts ---
    result = {"test": test_metrics, "history": history, "checkpoint": ckpt_path,
              "y_true": y_true, "y_prob": y_prob}
    results_path = os.path.join(save_dir, f"{name}_results.json")
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    return result
