import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import optim

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.Model_structure import RT_est
from utils.checkpoints import save_checkpoint
from utils.paths import NetPaths

if __package__ in (None, ""):
    from awssdr_repro import config as cfg
    from awssdr_repro.dataset_io import make_loader
    from awssdr_repro.eval_t30 import evaluate
else:
    from . import config as cfg
    from .dataset_io import make_loader
    from .eval_t30 import evaluate


def write_loss_curve(log_csv: Path, out_png: Path):
    df = pd.read_csv(log_csv)
    if df.empty:
        return
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(df["epoch"], df["train_loss"], marker="o", label="train MSE")
    if "dev_mse" in df.columns:
        dev = df.dropna(subset=["dev_mse"])
        if not dev.empty:
            ax.plot(dev["epoch"], dev["dev_mse"], marker="o", label="validation MSE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("AWSSDR T30 training curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def weights_init_normal(m):
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0)
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0)


def set_reproducibility(seed: int, deterministic: bool):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = torch.cuda.is_available()
        torch.use_deterministic_algorithms(False)


def train(args):
    set_reproducibility(args.seed, args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    paths = NetPaths(args.checkpoint_id)
    model = RT_est(num_channels=cfg.N_MELS, fc_dim=args.fc_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.apply(weights_init_normal)
    loss_fn = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_lr)
    loader = make_loader(
        args.train,
        args.batch_size,
        shuffle_batches=True,
        num_workers=args.num_workers,
    )
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        running_loss = 0.0
        pending = 0
        model.train()
        optimizer.zero_grad()
        for batch_idx, (dr, target) in enumerate(loader, 1):
            dr = dr[:, :, -cfg.N_MELS :].to(device, torch.float32)
            target = target.to(device, torch.float32).view(-1, 1)
            pred, *_ = model(dr)
            loss = loss_fn(pred, target)
            loss.backward()
            pending += 1
            if pending == args.update_period:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if torch.isnan(grad_norm):
                    raise FloatingPointError("gradient norm is NaN")
                optimizer.step()
                optimizer.zero_grad()
                pending = 0
            running_loss += float(loss.item())
            if batch_idx % args.log_every == 0 or batch_idx == len(loader):
                print(
                    f"epoch={epoch}/{args.epochs} batch={batch_idx}/{len(loader)} "
                    f"loss={running_loss / batch_idx:.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.6g} "
                    f"speed={batch_idx / max(time.time() - start, 1e-9):.2f} batch/s"
                )

        if pending:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if torch.isnan(grad_norm):
                raise FloatingPointError("gradient norm is NaN")
            optimizer.step()
            optimizer.zero_grad()

        train_loss = running_loss / max(len(loader), 1)
        save_checkpoint(paths, model, optimizer, is_silent=True)
        if epoch > args.decay_epoch:
            scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss}

        if args.dev and epoch % args.eval_check == 0:
            _, dev_metrics = evaluate(
                args.dev,
                paths.checkpoints,
                cfg.REPORTS_DIR / "dev_predictions.csv",
                cfg.REPORTS_DIR / "dev_metrics.json",
            )
            row.update({f"dev_{k}": v for k, v in dev_metrics["overall"].items()})
        log_rows.append(row)
        cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        log_csv = cfg.REPORTS_DIR / "train_log.csv"
        pd.DataFrame(log_rows).to_csv(log_csv, index=False)
        write_loss_curve(log_csv, cfg.REPORTS_DIR / "loss_curve.png")
    return model


def main():
    defaults = cfg.TRAINING
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=cfg.TRAIN_FEATURE_DIR)
    parser.add_argument("--dev", type=Path, default=cfg.ACE_DEV_FEATURE_DIR)
    parser.add_argument("--checkpoint-id", default=cfg.CHECKPOINT_ID)
    parser.add_argument("--epochs", type=int, default=defaults["max_epochs"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--lr", type=float, default=defaults["lr"])
    parser.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--fc-dim", type=int, default=defaults["fc_dim"])
    parser.add_argument("--decay-epoch", type=int, default=defaults["decay_epoch"])
    parser.add_argument("--decay-lr", type=float, default=defaults["decay_lr"])
    parser.add_argument("--eval-check", type=int, default=defaults["eval_check"])
    parser.add_argument("--update-period", type=int, default=defaults["update_period"])
    parser.add_argument("--grad-clip", type=float, default=defaults["grad_clip"])
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=defaults["num_workers"])
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="enable strict deterministic algorithms; may require CUBLAS_WORKSPACE_CONFIG on CUDA",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
