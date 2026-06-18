"""
train.py
Train the T60 CNN (Gamper & Tashev 2018) on the precomputed feature
shards in data/features/{train,val}/.

- Adam (lr 1e-3), MSE loss, batch size 128
- Early stopping on validation MSE (patience 30), max 500 epochs
- Saves best checkpoint to runs/best_model.pt and the training curve
  to runs/training_curve.png

Run:  python src/train.py
"""

import os
import glob
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import T60CNN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR = os.path.join(ROOT, "data", "features")
RUNS_DIR = os.path.join(ROOT, "runs")

SEED = 42
BATCH_SIZE = 128
LR = 1e-3
MAX_EPOCHS = 300
PATIENCE = 30


class ShardDataset(Dataset):
    """Loads every .npz shard of a split into RAM (float16 storage,
    cast to float32 per item)."""

    def __init__(self, split):
        paths = sorted(glob.glob(os.path.join(FEAT_DIR, split, "*.npz")))
        if not paths:
            raise FileNotFoundError(f"no shards in {FEAT_DIR}/{split}")
        Xs, ys = [], []
        for p in paths:
            d = np.load(p)
            if d["y"].shape[0]:
                Xs.append(d["X"])
                ys.append(d["y"])
        self.X = np.concatenate(Xs)            # (N, 21, T) float16
        self.y = np.concatenate(ys).astype(np.float32)
        print(f"{split}: {len(self.y)} chunks, "
              f"{self.X.nbytes / 1e9:.2f} GB in RAM")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i].astype(np.float32)).unsqueeze(0)
        return x, self.y[i]


def evaluate(model, loader, device):
    model.eval()
    se, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device))
            se += ((pred.cpu() - y) ** 2).sum().item()
            n += len(y)
    return se / n


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(RUNS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ds = ShardDataset("train")
    val_ds = ShardDataset("val")

    # Trivial baseline: always predict the mean of the training labels
    base_mse = float(np.mean((val_ds.y - train_ds.y.mean()) ** 2))
    print(f"baseline (predict train mean {train_ds.y.mean():.3f}s): "
          f"val MSE = {base_mse:.4f}")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=6, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=6, pin_memory=True)

    model = T60CNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val, best_epoch = float("inf"), -1
    hist_train, hist_val = [], []
    t0 = time.time()

    for epoch in range(MAX_EPOCHS):
        model.train()
        se, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            se += loss.item() * len(y)
            n += len(y)
        train_mse = se / n
        val_mse = evaluate(model, val_dl, device)
        hist_train.append(train_mse)
        hist_val.append(val_mse)

        marker = ""
        if val_mse < best_val:
            best_val, best_epoch = val_mse, epoch
            torch.save({"model": model.state_dict(),
                        "epoch": epoch, "val_mse": val_mse},
                       os.path.join(RUNS_DIR, "best_model.pt"))
            marker = " *"
        print(f"epoch {epoch + 1:3d} | train MSE {train_mse:.4f} | "
              f"val MSE {val_mse:.4f}{marker} | "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)

        if epoch - best_epoch >= PATIENCE:
            print(f"early stop: no val improvement for {PATIENCE} epochs")
            break

    print(f"\nbest val MSE {best_val:.4f} @ epoch {best_epoch + 1} "
          f"(baseline {base_mse:.4f})")

    plt.figure(figsize=(8, 5))
    plt.semilogy(hist_train, label="train MSE")
    plt.semilogy(hist_val, label="val MSE")
    plt.axhline(base_mse, color="gray", ls="--", label="predict-mean baseline")
    plt.xlabel("epoch")
    plt.ylabel("MSE [s^2]")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RUNS_DIR, "training_curve.png"), dpi=150)
    print(f"saved {os.path.join(RUNS_DIR, 'training_curve.png')}")


if __name__ == "__main__":
    main()
