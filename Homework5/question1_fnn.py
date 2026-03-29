import argparse

import numpy as np
import torch
from torch import nn

from mnist_nn_utils import (
    ensure_dir,
    get_device,
    load_mnist,
    save_json,
    set_seed,
    train_and_evaluate,
)


class FNN(nn.Module):
    def __init__(self, hidden_layers, dropout: float):
        super().__init__()
        layers = []
        in_dim = 28 * 28
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Homework5 Q1 FNN tuner")
    parser.add_argument("--quick", action="store_true", help="Run fewer configs/epochs for smoke testing")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    output_dir = ensure_dir("Homework5/results")

    configs = [
        {"hidden_layers": [128, 64], "dropout": 0.0, "batch_size": 128, "lr": 1e-3},
        {"hidden_layers": [128, 64], "dropout": 0.2, "batch_size": 128, "lr": 1e-3},
        {"hidden_layers": [256, 128], "dropout": 0.2, "batch_size": 128, "lr": 1e-3},
        {"hidden_layers": [256, 128, 64], "dropout": 0.3, "batch_size": 128, "lr": 8e-4},
        {"hidden_layers": [512, 256, 128], "dropout": 0.3, "batch_size": 64, "lr": 8e-4},
    ]

    if args.quick:
        configs = configs[:2]

    epochs = args.epochs
    patience = args.patience
    weight_decay = 1e-4

    all_results = []
    best = None

    print(f"Tuning {len(configs)} FNN configurations...")
    for i, cfg in enumerate(configs, start=1):
        print(f"[{i}/{len(configs)}] {cfg}")
        train_loader, val_loader, test_loader = load_mnist(
            data_root="Homework5/data", batch_size=cfg["batch_size"], for_cnn=False, seed=42
        )
        model = FNN(hidden_layers=cfg["hidden_layers"], dropout=cfg["dropout"])
        result = train_and_evaluate(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            lr=cfg["lr"],
            weight_decay=weight_decay,
            epochs=epochs,
            patience=patience,
            device=device,
        )

        item = {
            "config": cfg,
            "best_val_acc": result.best_val_acc,
            "best_epoch": result.best_epoch,
            "test_acc": result.test_acc,
            "train_time_sec": result.train_time_sec,
        }
        all_results.append(item)
        print(
            f"    val={result.best_val_acc:.4f} test={result.test_acc:.4f} "
            f"epoch={result.best_epoch} time={result.train_time_sec:.1f}s"
        )

        if best is None or result.test_acc > best["result"].test_acc:
            best = {"cfg": cfg, "result": result, "model": model}

    ranked = sorted(all_results, key=lambda x: x["test_acc"], reverse=True)
    summary = {
        "model": "FNN",
        "configs_tested": len(configs),
        "epochs": epochs,
        "patience": patience,
        "weight_decay": weight_decay,
        "best": {
            "config": best["cfg"],
            "test_acc": best["result"].test_acc,
            "best_val_acc": best["result"].best_val_acc,
            "best_epoch": best["result"].best_epoch,
            "train_time_sec": best["result"].train_time_sec,
        },
        "top_configs": ranked,
    }

    save_json(output_dir / "fnn_results.json", summary)
    np.save(output_dir / "fnn_confusion_matrix.npy", best["result"].confusion_matrix)
    torch.save(best["model"].state_dict(), output_dir / "best_fnn_state.pt")

    print("Saved:")
    print("  Homework5/results/fnn_results.json")
    print("  Homework5/results/fnn_confusion_matrix.npy")
    print("  Homework5/results/best_fnn_state.pt")


if __name__ == "__main__":
    main()
