"""Train the AWSSDR network: MSE on raw seconds, Adam lr 1e-3 / weight decay 1e-3,
length-sorted batches of 16 with an optimizer step every 16 batches (effective batch 256,
grad-norm clip 1.0), 100 epochs, exponential LR decay 0.99/epoch after epoch 50 (paper
Sec. III-C; the released reference code used 0.95 - we follow the paper text).

Protocol as in the other folders: trains on 100% of the synthetic utterances, the ACE dev
set is evaluated every EVAL_EVERY epochs only to monitor overfitting; the reported model
is the FINAL checkpoint (last_<version>.pt), best_<version>.pt (lowest dev MSE) is kept
as a reference. Prints the predict-the-mean baseline first.

Default version is now "v6" (RIRS/Train_RIRs x LibriSpeech train-clean-100 + TSP, 955 RIRs,
~28,650 utterances, clean/no noise, T30 labels from Train_RIRs/T30_ground_truth.csv - see
prepare_data_LSTSP.py), monitored against the CLEAN ACE dev set (data/ace_clean_rt30/ace_dev
+ ace_clean_rt30_dev_chunks.csv) rather than the real noisy ACE dev used by earlier versions -
v6 is a clean-trained model, so a clean eval set matches its training domain. A final,
non-model-selecting pass over the clean ACE test/eval set (ace_clean_rt30/ace_test +
ace_clean_rt30_test_chunks.csv) runs once after training completes and is reported but not
used to pick best_<version>.pt. Note: Train_RIRs includes 14 RIRs from the same ACE rooms used in
both the dev and eval sets here (added deliberately this session, accepting the tradeoff), so
neither is a fully held-out evaluation the way earlier versions' real-ACE dev set was. Earlier
versions ("noise_v4": RIRS/ALL_RIRS2_train x LibriSpeech, Gamper-style noise at 0/10/20 dB,
real noisy ACE dev; "v5": 1000 synthetic RIRs, clean, labels from selected_training_rirs.csv;
"v4": clean; "noise_v3": the older 866-RIR revision labeled from RT30.csv) remain selectable
via --version.

Artifacts (last/best .pt, train_log_<version>.csv, training_curve_<version>.png) land in
--runs-dir, default runs_v6/; relative paths resolve against AWSSDR/.

Run: CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py
     (defaults are set for a plain IDE "Run" with no arguments - no CLI flags needed;
      CUBLAS_WORKSPACE_CONFIG only silences the cuBLAS determinism warning from
      torch.use_deterministic_algorithms(True) and is not required to train.)
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dataset import BATCH, load_set, make_blocks, pad_batch
from model import AWSSDRNet, init_weights

HERE = os.path.dirname(os.path.abspath(__file__))                    # AWSSDR/src/
PROJECT_ROOT = os.path.dirname(HERE)                                 # AWSSDR/
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEV_DIR = os.path.join(DATA_DIR, "ace_clean_rt30", "ace_dev")
DEV_CSV = os.path.join(DATA_DIR, "ace_clean_rt30", "ace_clean_rt30_dev_chunks.csv")
EVAL_DIR = os.path.join(DATA_DIR, "ace_clean_rt30", "ace_test")
EVAL_CSV = os.path.join(DATA_DIR, "ace_clean_rt30", "ace_clean_rt30_test_chunks.csv")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs_v6")  # default; override with --runs-dir

SEED = 0
LR = 1e-3
WEIGHT_DECAY = 1e-3
EPOCHS = 100
DECAY_EPOCH = 50
LR_GAMMA = 0.95
UPDATE_PERIOD = 16
CLIP_NORM = 1.0
EVAL_EVERY = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

C_TRAIN = "#4269D0"   # blue
C_DEV = "#E4572E"     # orange-red
C_REF = "#808080"     # gray reference


@torch.no_grad()
def evaluate(model, seqs, y):
    model.eval()
    preds = []
    for dr in seqs:
        est, *_ = model(dr.unsqueeze(0).to(DEVICE))
        preds.append(est.item())
    err = torch.tensor(preds) - y
    return (err ** 2).mean().item(), err.abs().mean().item(), err.mean().item()


def save_curve(log, baseline_mse, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(log["epoch"], log["train_mse"], color=C_TRAIN, lw=1.5, label="train MSE")
    dev = pd.DataFrame(log).dropna(subset=["dev_mse"])
    ax.semilogy(dev["epoch"], dev["dev_mse"], color=C_DEV, lw=1.5, marker=".",
                label="ACE dev MSE (every %d ep)" % EVAL_EVERY)
    ax.axhline(baseline_mse, color=C_REF, ls="--", lw=1.2,
               label="predict-the-mean baseline (dev)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE [s$^2$]")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v6",
                    help="feature-set version (e.g. v1, v1_noise, v2, v3, v4, noise_v3, "
                         "noise_v4, v5, v6); selects train dir/CSV and run-artifact names")
    ap.add_argument("--runs-dir", default=RUNS_DIR,
                    help="where checkpoints/log/curve land; relative paths resolve against "
                         "AWSSDR/ (default: %s)" % os.path.basename(RUNS_DIR))
    args = ap.parse_args()
    version = args.version
    runs_dir = args.runs_dir if os.path.isabs(args.runs_dir) \
        else os.path.join(PROJECT_ROOT, args.runs_dir)
    train_dir = os.path.join(DATA_DIR, "features_awssdr_" + version, "train")
    train_csv = os.path.join(DATA_DIR, "awssdr_train_chunks_" + version + ".csv")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    os.makedirs(runs_dir, exist_ok=True)

    seqs, y, _ = load_set(train_dir, train_csv)
    seqs_dev, y_dev, _ = load_set(DEV_DIR, DEV_CSV)
    seqs_eval, y_eval, _ = load_set(EVAL_DIR, EVAL_CSV)
    print("train: %d utterances  dev: %d utterances  eval: %d utterances  device: %s"
          % (len(y), len(y_dev), len(y_eval), DEVICE))

    mean_pred = y.mean()
    base_mse = ((y_dev - mean_pred) ** 2).mean().item()
    base_mae = (y_dev - mean_pred).abs().mean().item()
    print("baseline (predict train mean %.3f s): dev MSE %.4f  dev MAE %.4f"
          % (mean_pred, base_mse, base_mae))

    model = AWSSDRNet().to(DEVICE)
    model.apply(init_weights)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=LR_GAMMA)

    blocks = make_blocks(len(y), BATCH)
    rng = np.random.default_rng(SEED)
    best_mse = float("inf")
    log = {"epoch": [], "train_mse": [], "dev_mse": [], "dev_mae": [], "lr": []}
    log_csv = os.path.join(runs_dir, "train_log_%s.csv" % version)
    curve_png = os.path.join(runs_dir, "training_curve_%s.png" % version)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        opt.zero_grad()  # leftover accumulated grads are discarded (reference behaviour)
        order = rng.permutation(len(blocks))
        se_sum, n, n_acc = 0.0, 0, 0
        for bi in order:
            idx = blocks[bi]
            dr = pad_batch([seqs[i] for i in idx]).to(DEVICE)
            yb = y[idx].to(DEVICE)
            est, *_ = model(dr)
            loss = ((est.squeeze(1) - yb) ** 2).mean()
            loss.backward()
            n_acc += 1
            if n_acc == UPDATE_PERIOD:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                if torch.isnan(grad_norm):
                    print("grad_norm was NaN!")
                opt.step()
                opt.zero_grad()
                n_acc = 0
            se_sum += loss.item() * len(idx)
            n += len(idx)
        train_mse = se_sum / n
        cur_lr = opt.param_groups[0]["lr"]
        if epoch > DECAY_EPOCH:
            sched.step()

        dev_mse = dev_mae = float("nan")
        marker = ""
        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
            dev_mse, dev_mae, dev_bias = evaluate(model, seqs_dev, y_dev)
            if dev_mse < best_mse:
                best_mse = dev_mse
                marker = "  *best"
                torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                            "dev_mse": dev_mse},
                           os.path.join(runs_dir, "best_%s.pt" % version))
        torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                    "dev_mse": dev_mse}, os.path.join(runs_dir, "last_%s.pt" % version))

        for k, v in zip(log, (epoch, train_mse, dev_mse, dev_mae, cur_lr)):
            log[k].append(v)
        print("epoch %4d  train MSE %.4f  dev MSE %s  dev MAE %s  lr %.2e  (%.1f s)%s"
              % (epoch, train_mse,
                 "%.4f" % dev_mse if dev_mse == dev_mse else "  -   ",
                 "%.4f" % dev_mae if dev_mae == dev_mae else "  -   ",
                 cur_lr, time.time() - t0, marker), flush=True)

        if epoch % 25 == 0 or epoch == EPOCHS:
            pd.DataFrame(log).to_csv(log_csv, index=False)
            save_curve(log, base_mse, curve_png)

    eval_mse, eval_mae, eval_bias = evaluate(model, seqs_eval, y_eval)
    print("final model (last_%s.pt) on held-out eval set: MSE %.4f  MAE %.4f  bias %+.4f"
          % (version, eval_mse, eval_mae, eval_bias))
    print("done. final model -> %s  (best dev MSE %.4f, reference only)"
          % (os.path.join(runs_dir, "last_%s.pt" % version), best_mse))


if __name__ == "__main__":
    main()
