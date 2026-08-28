"""Plot AWSSDR training logs with a readable post-initialisation zoom.

The first epoch can have a much larger MSE than the remaining epochs.  This
script keeps a full-history logarithmic panel and adds a linear panel that,
by default, starts at epoch 2 so the useful training dynamics are visible.

Example:
    python -m awssdr_repro.plot_loss_curve

To include every epoch in the zoom panel:
    python -m awssdr_repro.plot_loss_curve --start-epoch 1
"""

import argparse
import os
from pathlib import Path

import pandas as pd


def plot_loss(log_path: Path, output_path: Path, start_epoch: int) -> None:
    # Import plotting libraries only when the script is actually used.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(log_path)
    required = {"epoch", "train_loss"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required log columns: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"Training log is empty: {log_path}")

    df = df.sort_values("epoch")
    zoom = df[df["epoch"] >= start_epoch]
    if zoom.empty:
        raise ValueError(f"No log rows at or after epoch {start_epoch}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 5),
                                            constrained_layout=True)

    # Full history: log scale prevents the initial spike from hiding the rest.
    ax_full.semilogy(df["epoch"], df["train_loss"], marker=".",
                     label="train MSE")
    if "dev_mse" in df.columns:
        dev = df.dropna(subset=["dev_mse"])
        if not dev.empty:
            ax_full.semilogy(dev["epoch"], dev["dev_mse"], marker="o",
                             label="ACE DEV MSE")
    ax_full.set_title("Full training history (log scale)")
    ax_full.set_xlabel("epoch")
    ax_full.set_ylabel("MSE [s²]")
    ax_full.grid(True, which="both", alpha=0.25)
    ax_full.legend()

    # Linear zoom: this is the view intended for assessing convergence.
    ax_zoom.plot(zoom["epoch"], zoom["train_loss"], marker=".",
                 label="train MSE")
    if "dev_mse" in zoom.columns:
        dev_zoom = zoom.dropna(subset=["dev_mse"])
        if not dev_zoom.empty:
            ax_zoom.plot(dev_zoom["epoch"], dev_zoom["dev_mse"], marker="o",
                         label="ACE DEV MSE")
    ax_zoom.set_title(f"Convergence view (epoch ≥ {start_epoch})")
    ax_zoom.set_xlabel("epoch")
    ax_zoom.set_ylabel("MSE [s²]")
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.legend()

    fig.suptitle("AWSSDR T30 training loss", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"wrote {output_path}")


def main() -> None:
    from . import config as cfg

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        type=Path,
        default=cfg.REPORTS_DIR / "train_log.csv",
        help="training CSV log",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=cfg.REPORTS_DIR / "loss_curve_zoomed.png",
        help="output PNG path",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=2,
        help="first epoch shown in the linear convergence panel",
    )
    args = parser.parse_args()
    plot_loss(args.log, args.out, args.start_epoch)


if __name__ == "__main__":
    main()
