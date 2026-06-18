"""
evaluate.py
Evaluate the trained T60 CNN on the train and val splits, reporting
the ACE Challenge metrics used in Gamper & Tashev 2018:

  - bias (mean estimation error)
  - MSE
  - Pearson correlation coefficient rho

Plots: 2D truth-vs-estimate histograms ("confusion matrices") and
error binned by ground-truth T60 (0.1 s bins) for train and val.

Run:  python src/evaluate.py
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


def predict_split(model, device, split="test"):
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

    # --- Metrics for train and val splits --------------------------------
    results = {}
    for split in ("train", "val"):
        y_true, y_pred, rir_rows = predict_split(model, device, split)
        bias, mse, rho = metrics(y_true, y_pred)
        rt = np.array([r[1] for r in rir_rows])
        rp = np.array([r[2] for r in rir_rows])
        b2, m2, r2 = metrics(rt, rp)
        results[split] = dict(y_true=y_true, y_pred=y_pred,
                              rir_rows=rir_rows,
                              bias=bias, mse=mse, rho=rho,
                              bias_rir=b2, mse_rir=m2, rho_rir=r2)
        print(f"\n{split.upper()} (per chunk, n={len(y_true)}):")
        print(f"  bias = {bias:+.4f} s | MSE = {mse:.4f} s^2 | rho = {rho:.3f}")
        print(f"{split.upper()} (per RIR avg, n={len(rt)}):")
        print(f"  bias = {b2:+.4f} s | MSE = {m2:.4f} s^2 | rho = {r2:.3f}")

    # --- Side-by-side confusion matrices: train | val --------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, split in zip(axes, ("train", "val")):
        r = results[split]
        title = (f"{split}  (n={len(r['y_true'])})\n"
                 f"bias={r['bias']:+.3f}  MSE={r['mse']:.4f}  ρ={r['rho']:.3f}")
        im = plot_confusion(ax, r["y_true"], r["y_pred"], title)
    fig.colorbar(im, ax=axes[-1], label="% of estimates per true T60 bin", shrink=0.8)
    fig.suptitle("Ground-truth vs estimated T60 (0.1 s bins)", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(RUNS_DIR, "confusion_train_val.png"), dpi=150,
                bbox_inches="tight")
    print(f"\nsaved confusion_train_val.png to {RUNS_DIR}")

    # --- Error-by-T60 boxplots: train and val ----------------------------
    bins = np.arange(0.1, 1.6, 0.1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    for ax, split in zip(axes, ("train", "val")):
        y_true = results[split]["y_true"]
        y_pred = results[split]["y_pred"]
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
        ax.set_title(f"Estimation error by T60 ({split})")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RUNS_DIR, "error_by_t60_train_val.png"), dpi=150)
    print(f"saved error_by_t60_train_val.png to {RUNS_DIR}")

    # --- Worst val RIRs --------------------------------------------------
    rir_rows = results["val"]["rir_rows"]
    rows = sorted(rir_rows, key=lambda r: abs(r[2] - r[1]), reverse=True)[:10]
    print("\nworst val RIRs (|utterance-avg error|):")
    for name, t, p in rows:
        print(f"  {name:45s} true {t:.3f}  est {p:.3f}  err {p - t:+.3f}")


if __name__ == "__main__":
    main()
