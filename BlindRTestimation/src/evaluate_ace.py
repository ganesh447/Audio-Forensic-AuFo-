"""
evaluate_ace.py
Evaluate the trained T60 CNN on the ACE evaluation set (data/features/test),
reporting the same ACE Challenge metrics used in Gamper & Tashev 2018:

  - bias (mean estimation error)
  - MSE
  - Pearson correlation coefficient rho

Plots: 2D truth-vs-estimate histogram ("confusion matrix") and
error binned by ground-truth T60 (0.1 s bins) for the ACE test set.

Run:  python src/evaluate_ace.py
"""

import os
import glob

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import T60CNN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR = os.path.join(ROOT, "data", "features")
RUNS_DIR = os.path.join(ROOT, "runs")
CKPT = os.path.join(RUNS_DIR, "best_model.pt")
BATCH = 256
SPLIT = "test"


def predict_split(model, device, split=SPLIT):
    """Per-chunk predictions plus per-RIR (utterance-averaged) results."""
    paths = sorted(glob.glob(os.path.join(FEAT_DIR, split, "*.npz")))
    y_true, y_pred, rir_rows = [], [], []
    with torch.no_grad():
        for p in paths:
            d = np.load(p)
            X, y = d["X"], d["y"]
            if not len(y):
                continue
            preds = []
            for i in range(0, len(y), BATCH):
                xb = torch.from_numpy(X[i:i + BATCH].astype(np.float32))
                xb = xb.unsqueeze(1).to(device)
                preds.append(model(xb).cpu().numpy())
            preds = np.concatenate(preds)
            y_true.append(y)
            y_pred.append(preds)
            rir_rows.append((os.path.basename(p), float(y[0]),
                             float(preds.mean())))
    return (np.concatenate(y_true), np.concatenate(y_pred), rir_rows)


def metrics(y_true, y_pred):
    err = y_pred - y_true
    bias = float(err.mean())
    mse = float((err ** 2).mean())
    rho = float(np.corrcoef(y_true, y_pred)[0, 1])
    return bias, mse, rho


def plot_confusion(ax, y_true, y_pred, title):
    """Row-normalised 2D histogram on a given Axes."""
    edges = np.arange(0.0, 1.65, 0.1)
    h, _, _ = np.histogram2d(y_true, y_pred, bins=[edges, edges])
    h = h / np.maximum(h.sum(axis=1, keepdims=True), 1) * 100
    im = ax.imshow(h.T, origin="lower", extent=[0, 1.5, 0, 1.5],
                   aspect="equal", cmap="viridis", vmin=0, vmax=100)
    ax.plot([0, 1.5], [0, 1.5], "w--", lw=0.8)
    ax.set_xlabel("ground truth T60 [s]")
    ax.set_ylabel("estimated T60 [s]")
    ax.set_title(title)
    return im


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CKPT, map_location=device, weights_only=True)
    model = T60CNN().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {CKPT} (epoch {ckpt['epoch'] + 1}, "
          f"val MSE {ckpt['val_mse']:.4f})")

    # --- Metrics for the ACE test split -----------------------------------
    y_true, y_pred, rir_rows = predict_split(model, device, SPLIT)
    bias, mse, rho = metrics(y_true, y_pred)
    rt = np.array([r[1] for r in rir_rows])
    rp = np.array([r[2] for r in rir_rows])
    b2, m2, r2 = metrics(rt, rp)
    print(f"\nACE {SPLIT.upper()} (per chunk, n={len(y_true)}):")
    print(f"  bias = {bias:+.4f} s | MSE = {mse:.4f} s^2 | rho = {rho:.3f}")
    print(f"ACE {SPLIT.upper()} (per file avg, n={len(rt)}):")
    print(f"  bias = {b2:+.4f} s | MSE = {m2:.4f} s^2 | rho = {r2:.3f}")

    # --- Confusion matrix --------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 5))
    title = (f"ACE {SPLIT}  (n={len(y_true)})\n"
             f"bias={bias:+.3f}  MSE={mse:.4f}  ρ={rho:.3f}")
    im = plot_confusion(ax, y_true, y_pred, title)
    fig.colorbar(im, ax=ax, label="% of estimates per true T60 bin", shrink=0.8)
    fig.suptitle("Ground-truth vs estimated T60 (0.1 s bins) -- ACE", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RUNS_DIR, "confusion_ace_test.png"), dpi=150,
                bbox_inches="tight")
    print(f"\nsaved confusion_ace_test.png to {RUNS_DIR}")

    # --- Error-by-T60 boxplot ------------------------------------------------
    bins = np.arange(0.1, 1.6, 0.1)
    fig, ax = plt.subplots(figsize=(7, 4))
    centers, box_data = [], []
    for lo in bins[:-1]:
        m = (y_true >= lo) & (y_true < lo + 0.1)
        if m.sum():
            centers.append(lo + 0.05)
            box_data.append(y_pred[m] - y_true[m])
    ax.boxplot(box_data, positions=centers, widths=0.06, showfliers=False)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(np.round(bins, 1))
    ax.set_xticklabels(np.round(bins, 1))
    ax.set_xlabel("ground truth T60 [s]")
    ax.set_ylabel("error [s]")
    ax.set_title("Estimation error by T60 (ACE test)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RUNS_DIR, "error_by_t60_ace_test.png"), dpi=150)
    print(f"saved error_by_t60_ace_test.png to {RUNS_DIR}")

    # --- Worst ACE files -----------------------------------------------------
    rows = sorted(rir_rows, key=lambda r: abs(r[2] - r[1]), reverse=True)[:10]
    print("\nworst ACE files (|file-avg error|):")
    for name, t, p in rows:
        print(f"  {name:55s} true {t:.3f}  est {p:.3f}  err {p - t:+.3f}")


if __name__ == "__main__":
    main()
