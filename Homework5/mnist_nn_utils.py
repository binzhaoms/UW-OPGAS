import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


@dataclass
class TrainResult:
    best_val_acc: float
    best_epoch: int
    test_acc: float
    train_time_sec: float
    history: dict
    confusion_matrix: np.ndarray


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_mnist(data_root: str, batch_size: int, for_cnn: bool, seed: int = 42):
    transform = transforms.ToTensor()
    train_full = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    g = torch.Generator().manual_seed(seed)
    train_len = 54000
    val_len = len(train_full) - train_len
    train_set, val_set = random_split(train_full, [train_len, val_len], generator=g)

    def collate_flat(batch):
        xs, ys = zip(*batch)
        x = torch.stack(xs).view(len(xs), -1)
        y = torch.tensor(ys, dtype=torch.long)
        return x, y

    collate = None if for_cnn else collate_flat

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader, test_loader


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    device: torch.device,
) -> TrainResult:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_state = None
    best_val_acc = -1.0
    best_epoch = -1
    stale_epochs = 0

    start = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * yb.size(0)
            train_correct += (logits.argmax(dim=1) == yb).sum().item()
            train_total += yb.size(0)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * yb.size(0)
                val_correct += (logits.argmax(dim=1) == yb).sum().item()
                val_total += yb.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            break

    train_time = time.time() - start

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(yb.numpy().tolist())

    y_true_arr = np.array(y_true, dtype=np.int64)
    y_pred_arr = np.array(y_pred, dtype=np.int64)
    test_acc = float((y_true_arr == y_pred_arr).mean())
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(10)))

    return TrainResult(
        best_val_acc=float(best_val_acc),
        best_epoch=int(best_epoch),
        test_acc=test_acc,
        train_time_sec=float(train_time),
        history=history,
        confusion_matrix=cm,
    )


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
