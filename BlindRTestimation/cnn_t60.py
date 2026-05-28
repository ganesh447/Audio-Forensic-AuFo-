"""
Blind T60 estimation — Gamper & Tashev IWAENC 2018
CNN implementation using PyTorch.
"""

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from scipy.stats import pearsonr
from tqdm import tqdm


# ── 1. Dataset ─────────────────────────────────────────────────────────────────

class T60Dataset(Dataset):

    def __init__(self, manifest_csv: Path, base_dir: Path):
        self.base_dir = base_dir
        self.samples  = []
        with open(manifest_csv, newline="") as f:
            for row in csv.DictReader(f):
                self.samples.append({
                    "path": base_dir / row["filename"],
                    "t30":  float(row["t30"]),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s    = self.samples[idx]
        feat = np.load(str(s["path"]))              # (21, 1999)
        feat = torch.from_numpy(feat).unsqueeze(0)  # (1, 21, 1999)
        t30  = torch.tensor(s["t30"], dtype=torch.float32)
        return feat, t30


# ── 2. Building block ──────────────────────────────────────────────────────────

class ConvBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int,
                 kernel: tuple, stride: tuple):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel,
                      stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── 3. Model ───────────────────────────────────────────────────────────────────

class CNNT60(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1, 5, (1, 10), (1, 2)),   # → (5, 21, 995)
            ConvBlock(5, 5, (1, 10), (1, 3)),   # → (5, 21, 329)
            ConvBlock(5, 5, (1, 11), (1, 3)),   # → (5, 21, 107)
            ConvBlock(5, 5, (1, 11), (1, 2)),   # → (5, 21,  49)
            ConvBlock(5, 5, (3,  8), (2, 2)),   # → (5, 10,  21)
            ConvBlock(5, 5, (4,  7), (2, 1)),   # → (5,  4,  15)
        )

        self.dropout = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten()
        self.dense   = nn.Linear(5 * 4 * 15, 1)   # 300 → 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)   # (B, 5, 4, 15)
        x = self.dropout(x)
        x = self.flatten(x)    # (B, 300)
        x = self.dense(x)      # (B, 1)
        return x.squeeze(1)    # (B,)


# ── 4. Trainer ─────────────────────────────────────────────────────────────────

class Trainer:

    def __init__(self, model: CNNT60,
                 train_loader: DataLoader,
                 val_loader:   DataLoader,
                 device:       torch.device,
                 lr:           float = 1e-3,
                 epochs:       int   = 500,
                 checkpoint:   Path  = Path("best_model.pt")):

        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.epochs       = epochs
        self.checkpoint   = checkpoint
        self.optimizer    = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion    = nn.MSELoss()
        self.best_val_mse = float("inf")
        self.use_amp      = device.type == "cuda"
        self.scaler       = GradScaler("cuda", enabled=self.use_amp)

    # ── private ────────────────────────────────────────────────────────────────

    def _train_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        for feat, t30 in self.train_loader:
            feat, t30 = feat.to(self.device), t30.to(self.device)
            self.optimizer.zero_grad()
            with autocast("cuda", enabled=self.use_amp):
                loss = self.criterion(self.model(feat), t30)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            running_loss += loss.item() * len(t30)
        return running_loss / len(self.train_loader.dataset)

    def _val_epoch(self) -> dict:
        self.model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for feat, t30 in self.val_loader:
                with autocast("cuda", enabled=self.use_amp):
                    pred = self.model(feat.to(self.device)).cpu().numpy()
                preds.extend(pred.tolist())
                targets.extend(t30.numpy().tolist())

        preds   = np.array(preds)
        targets = np.array(targets)
        r, _    = pearsonr(preds, targets)
        return {
            "mse":       float(np.mean((preds - targets) ** 2)),
            "bias":      float(np.mean(preds - targets)),
            "pearson_r": float(r),
        }

    # ── public ─────────────────────────────────────────────────────────────────

    def train(self):
        print(f"Training on {self.device} for {self.epochs} epochs  |  AMP: {self.use_amp}\n")
        epoch_bar = tqdm(range(1, self.epochs + 1), desc="Training", unit="epoch")
        for epoch in epoch_bar:
            train_mse = self._train_epoch()
            metrics   = self._val_epoch()

            if metrics["mse"] < self.best_val_mse:
                self.best_val_mse = metrics["mse"]
                torch.save(self.model.state_dict(), self.checkpoint)

            epoch_bar.set_postfix({
                "train_mse": f"{train_mse:.4f}",
                "val_mse":   f"{metrics['mse']:.4f}",
                "bias":      f"{metrics['bias']:+.4f}",
                "r":         f"{metrics['pearson_r']:.4f}",
            })

            if epoch % 10 == 0 or epoch == 1:
                tqdm.write(
                    f"Epoch {epoch:5d}/{self.epochs} | "
                    f"Train MSE {train_mse:.4f} | "
                    f"Val MSE {metrics['mse']:.4f} | "
                    f"Bias {metrics['bias']:+.4f} | "
                    f"r {metrics['pearson_r']:.4f}"
                )

        print(f"\nBest val MSE : {self.best_val_mse:.4f}")
        print(f"Checkpoint   : {self.checkpoint}")

    def evaluate(self):
        self.model.load_state_dict(
            torch.load(self.checkpoint, map_location=self.device, weights_only=True)
        )
        m = self._val_epoch()
        print("\n── Evaluation (best checkpoint) ──────────")
        print(f"  MSE       : {m['mse']:.4f}")
        print(f"  Bias      : {m['bias']:+.4f}")
        print(f"  Pearson r : {m['pearson_r']:.4f}")
        return m


# ── 5. Entry point ─────────────────────────────────────────────────────────────

def main():
    BASE_DIR         = Path(__file__).resolve().parent
    TRAIN_MANIFEST   = BASE_DIR / "train_manifest.csv"
    VAL_MANIFEST     = BASE_DIR / "val_manifest.csv"
    CHECKPOINT       = BASE_DIR / "best_model.pt"

    # ── diagnostics ────────────────────────────────────────────────────────────
    print(f"PyTorch version  : {torch.__version__}")
    print(f"CUDA available   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU              : {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device     : {DEVICE}\n")

    # --- data ---
    train_ds = T60Dataset(TRAIN_MANIFEST, BASE_DIR)
    val_ds   = T60Dataset(VAL_MANIFEST,   BASE_DIR)
    print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=5, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=5, persistent_workers=True)

    # --- model ---
    model    = CNNT60()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params}")

    # --- train ---
    trainer = Trainer(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        device       = DEVICE,
        lr           = 1e-3,
        epochs       = 100,
        checkpoint   = CHECKPOINT,
    )
    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    main()
