
"""Wohnung counterpart to testbatch_inference.py: run T30 inference on the top-level
ears_concatenated BK_Rec recording (one long, all-EARS-stimuli-in-one-file capture per
position) for every Wohnung2 room folder that has DIN ground truth, and write one scatter
plot plus one ground-truth/estimate CSV per room/checkpoint into
AWSSDR/labs/wohnung_inference/.
"""

import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import librosa
from scipy.stats import pearsonr

from features import FS, MIN_SAMPLES, compute_decay_rates
from model import AWSSDRNet

HERE = os.path.dirname(os.path.abspath(__file__))                    # AWSSDR/src/
PROJECT_ROOT = os.path.dirname(HERE)                                 # AWSSDR/
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WOHNUNG_DIR = os.path.join(PROJECT_ROOT, "wohnung")
CKPTS = [os.path.join(PROJECT_ROOT, "runs_v6", "last_v6.pt")]
OUT_DIR = os.path.join(PROJECT_ROOT, "labs", "wohnung_inference")

CHANNEL = 0        # ch1 of the B&K multichannel recordings
DEVICE_TAG = "BK_Rec"  # only device requested for this pass
POS_RE = re.compile(r"(w\d+_r\d+)")

# same rationale as evaluate_lab.py/inference.py: real 16-bit-quantized wavs
# leave exact-zero runs in the reverb tail; dither below the quantization
# floor before feature extraction.
DITHER_STD = 1e-5
_rng = np.random.default_rng(0)

C_PTS = "#4269D0"     # estimate points (blue)
C_REF = "#808080"


def position_of(name):
    m = POS_RE.match(os.path.basename(name))
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Room folder / recording discovery
# ---------------------------------------------------------------------------

def find_din_json(room_name):
    """-> (din_json_path, raw_folder_name) for room_name's "*_DIN_data"
    sidecar subfolder (e.g. cut_stimuli_DIN_data), or (None, None) if the
    room has no DIN ground truth generated yet. The ground truth inside is
    derived from the ESS sweep recordings (an RIR analysis), so it is a
    position/device property independent of which stimulus set - cut_stimuli
    or ears_concatenated - is being run through the model."""
    room_path = os.path.join(WOHNUNG_DIR, room_name)
    if not os.path.isdir(room_path):
        return None, None
    for entry in sorted(os.listdir(room_path)):
        if not entry.endswith("_DIN_data"):
            continue
        candidate = os.path.join(room_path, entry, "rir_analysis_results.json")
        if os.path.isfile(candidate):
            return candidate, entry[: -len("_DIN_data")]
    return None, None


def wohnung_rooms():
    """-> sorted list of Wohnung2 room folder names that have a matching
    "*_DIN_data" sidecar with rir_analysis_results.json."""
    out = []
    for entry in sorted(os.listdir(WOHNUNG_DIR)):
        path = os.path.join(WOHNUNG_DIR, entry)
        if not os.path.isdir(path):
            continue
        din_json, _ = find_din_json(entry)
        if din_json:
            out.append(entry)
    return out


def gt_positions_from_din(din_json):
    """-> set of positions among DIN entries recorded on any BK_* device."""
    with open(din_json) as f:
        din = json.load(f)
    return {position_of(e["recording_name"]) for e in din if "BK" in e["recording_name"]} - {None}


def find_ears_concatenated_bk(room_path):
    """-> {position: wav_path} for every top-level ears_concatenated BK_Rec
    recording in room_path - the long (~3-5 min) single capture per position
    that already contains every EARS stimulus for that position back-to-back,
    as opposed to the individual per-clip cuts under cut_stimuli/."""
    wavs = {}
    for path in sorted(glob.glob(os.path.join(room_path, "w*_ears_concatenated_*_%s_*.wav" % DEVICE_TAG))):
        pos = position_of(path)
        if pos:
            wavs[pos] = path
    return wavs


def build_ground_truth(din_json, device_tag):
    """-> DataFrame indexed by position: t30_gt, from the DIN JSON, restricted
    to recordings from `device_tag`."""
    def _num(v):  # some entries store T60 as the literal string "nan" rather than a float
        return v if isinstance(v, (int, float)) else float("nan")

    with open(din_json) as f:
        din = json.load(f)
    rows = []
    for e in din:
        pos = position_of(e["recording_name"])
        if pos and device_tag in e["recording_name"]:
            ch = e["channels"]["ch%d" % (CHANNEL + 1)]
            rows.append({"position": pos, "t30_gt": _num(ch["T30"])})
    return pd.DataFrame(rows).set_index("position").sort_index()


# ---------------------------------------------------------------------------
# Inference on reverberant speech
# ---------------------------------------------------------------------------

def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model = AWSSDRNet().to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print("loaded %s (epoch %d, dev MSE %s)"
          % (ckpt_path, ckpt["epoch"],
             "%.4f" % ckpt["dev_mse"] if ckpt["dev_mse"] == ckpt["dev_mse"] else "n/a"))
    return model


@torch.no_grad()
def estimate(model, path):
    x, fs_in = sf.read(path, dtype="float32", always_2d=True)
    x = librosa.resample(x[:, CHANNEL].astype(np.float64), orig_sr=fs_in,
                         target_sr=FS).astype(np.float32)
    x = x / (np.max(np.abs(x)) + 1e-12)
    x = x + _rng.normal(0.0, DITHER_STD, len(x)).astype(x.dtype)
    if len(x) < MIN_SAMPLES:
        return float("nan")
    dr = torch.from_numpy(compute_decay_rates(x).T).unsqueeze(0).to(DEVICE)
    est, *_ = model(dr)
    return est.item()


# ---------------------------------------------------------------------------
# Reporting (metric conventions from evaluate.py / evaluate_lab.py)
# ---------------------------------------------------------------------------

def metrics(y_true, y_est):
    err = np.asarray(y_est) - np.asarray(y_true)
    rho = pearsonr(y_true, y_est)[0] if len(np.unique(y_true)) > 1 else float("nan")
    return {"bias": err.mean(), "mse": (err ** 2).mean(), "rho": rho}


def report(df, name):
    m = metrics(df["t30_gt"], df["t60_est"])
    print("%s  %-24s (n=%d): bias %+.4f  MSE %.4f  rho %.3f"
          % (name, "all positions", len(df), m["bias"], m["mse"], m["rho"]))


def plot_scatter(df, path, title):
    lim = (0, max(1.5, df["t30_gt"].max(), df["t60_est"].max()) + 0.1)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(lim, lim, color=C_REF, ls="--", lw=1.2)
    ax.scatter(df["t30_gt"], df["t60_est"], s=45, color=C_PTS, marker="D",
               edgecolors="white", linewidths=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Ground truth T30 [s]")
    ax.set_ylabel("Estimated T30 [s]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-room driver
# ---------------------------------------------------------------------------

def process_room(room_name, models):
    din_json, _ = find_din_json(room_name)
    room_path = os.path.join(WOHNUNG_DIR, room_name)

    gt_positions = gt_positions_from_din(din_json)
    if not gt_positions:
        print("  no BK-tagged ground truth, skipping")
        return

    wavs = find_ears_concatenated_bk(room_path)
    if not wavs:
        print("  no top-level ears_concatenated %s recordings found, skipping" % DEVICE_TAG)
        return

    gt = build_ground_truth(din_json, DEVICE_TAG)
    missing = sorted(set(gt.index) ^ set(wavs))
    if missing:
        print("  WARNING speech/RIR position mismatch: %s (using intersection)" % missing)
        common = sorted(set(gt.index) & set(wavs))
        gt = gt.loc[common]
        wavs = {p: wavs[p] for p in common}

    room_tag = re.sub(r"[^a-z0-9]+", "_", room_name.lower()).strip("_")
    print("\n=== %s === (%d positions, device=%s, ears_concatenated)"
          % (room_name, len(gt), DEVICE_TAG))

    for ckpt_path, model in models.items():
        ckpt_tag = os.path.splitext(os.path.basename(ckpt_path))[0]
        df = gt.copy()
        df["t60_est"] = [estimate(model, wavs[pos]) for pos in df.index]
        df["err"] = df["t60_est"] - df["t30_gt"]

        print(df[["t30_gt", "t60_est", "err"]]
              .to_string(float_format=lambda v: "%.3f" % v))
        report(df, "%s/%s (ears_concatenated)" % (room_tag, ckpt_tag))

        csv_path = os.path.join(OUT_DIR, "lab_results_ears_%s_%s.csv" % (room_tag, ckpt_tag))
        df.round(3).to_csv(csv_path)
        png_path = os.path.join(OUT_DIR, "lab_scatter_ears_%s_%s.png" % (room_tag, ckpt_tag))
        plot_scatter(df, png_path, room_name.replace("_", " ") + " (ears concatenated)")
        print("  wrote %s and %s" % (csv_path, png_path))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    models = {ckpt: load_model(ckpt) for ckpt in CKPTS}

    rooms = wohnung_rooms()
    print("found %d wohnung rooms: %s" % (len(rooms), rooms))

    for room_name in rooms:
        try:
            process_room(room_name, models)
        except Exception as e:
            print("  ERROR processing %s: %s" % (room_name, e))


if __name__ == "__main__":
    main()
