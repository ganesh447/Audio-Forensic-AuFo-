"""Generate Stage-A (clean) AWSSDR training features: convolve speech (EARS + TSP) with
the 1176 measured RIRs of training_rirs_16khz_2_fixed, compute the spectral decay rate
sequence per WHOLE utterance (no chunking - AWSSDR is sequence-level), save one .npz per
utterance (DR float32 (40, N), t60).

Targets are the Karjalainen ground-truth T60s (training_rirs_16khz_2_fixed_karjalainen_t60.csv),
matching the paper's labeling method. K=9 utterances per RIR ~= one pass of the paper's
full cross (538 RIRs x 6 noises x 3 SNRs -> here 1176 x 9 = 10,584 clean utterances).
Utterances are capped at MAX_S seconds to bound batch padding.

Usage: python prepare_data.py [--limit N]   (--limit: only the first N RIRs, smoke test)
"""

import argparse
import os
import time

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import fftconvolve

from features import FS, MIN_SAMPLES, compute_decay_rates

HERE = os.path.dirname(os.path.abspath(__file__))                    # AWSSDR/
ROOT = os.path.dirname(HERE)                                         # project root
RIR_DIR = os.path.join(ROOT, "RIRS", "training_rirs_16khz_2_fixed")
RIR_CSV = os.path.join(ROOT, "RIRS", "training_rirs_16khz_2_fixed_karjalainen_t60.csv")
EARS_DIR = os.path.join(ROOT, "Speech", "EARS")
TSP_DIR = os.path.join(ROOT, "Speech", "TSP", "16k")
EARS_PREFIXES = ("emo_", "rainbow_", "sentences_", "freeform_")

VERSION = "v1"
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(DATA_DIR, "features_awssdr_" + VERSION, "train")
INDEX_CSV = os.path.join(DATA_DIR, "awssdr_train_chunks_" + VERSION + ".csv")

SEED = 42
K_UTT_PER_RIR = 9
MAX_S = 15.0

_speech_cache = {}


def build_speech_pool():
    pool = []
    for spk in sorted(os.listdir(EARS_DIR)):
        d = os.path.join(EARS_DIR, spk)
        pool += [os.path.join(d, f) for f in sorted(os.listdir(d))
                 if f.endswith(".wav") and f.startswith(EARS_PREFIXES)]
    for spk in sorted(os.listdir(TSP_DIR)):
        d = os.path.join(TSP_DIR, spk)
        if not os.path.isdir(d):  # Licence.txt / Stats.txt live next to the speaker dirs
            continue
        pool += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".wav")]
    return pool


def load_speech(path):
    if path not in _speech_cache:
        x, _ = librosa.load(path, sr=FS)
        _speech_cache[path] = x
    return _speech_cache[path]


def load_rir(path):
    r, sr = sf.read(path, always_2d=True)
    assert sr == FS, (path, sr)
    r = r[:, 0]
    return r / (np.max(np.abs(r)) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N RIRs (smoke)")
    args = ap.parse_args()

    df = pd.read_csv(RIR_CSV)
    df["basename"] = df["filename"].map(os.path.basename)
    df = df.sort_values("basename").reset_index(drop=True)
    if args.limit:
        df = df.iloc[:args.limit]

    pool = build_speech_pool()
    print("RIRs: %d  speech pool: %d files  K=%d" % (len(df), len(pool), K_UTT_PER_RIR))
    os.makedirs(OUT_DIR, exist_ok=True)

    t0 = time.time()
    n_files, n_capped = 0, 0
    for i, row in df.iterrows():
        stem = os.path.splitext(row["basename"])[0]
        if os.path.exists(os.path.join(OUT_DIR, "%s__u%d.npz" % (stem, K_UTT_PER_RIR - 1))):
            continue  # RIR already fully processed (skip-if-exists; delete shards to regen)
        rir = load_rir(os.path.join(RIR_DIR, row["basename"]))
        rng = np.random.default_rng([SEED, i])
        utts = rng.choice(len(pool), size=K_UTT_PER_RIR, replace=False)

        for j, u in enumerate(utts):
            speech = load_speech(pool[u])
            rev = fftconvolve(speech, rir)  # full tail kept: the decay is the signal here
            rev = rev / (np.max(np.abs(rev)) + 1e-12)
            if len(rev) > int(MAX_S * FS):
                rev = rev[:int(MAX_S * FS)]
                n_capped += 1
            assert len(rev) >= MIN_SAMPLES, (pool[u], len(rev))
            dr = compute_decay_rates(rev)
            np.savez(os.path.join(OUT_DIR, "%s__u%d.npz" % (stem, j)),
                     DR=dr, t60=np.float32(row["T60_s"]))
            n_files += 1
            if i == 0 and j == 0:
                sf.write(os.path.join(DATA_DIR, "debug_clean_awssdr.wav"), speech, FS)
                sf.write(os.path.join(DATA_DIR, "debug_reverb_awssdr.wav"), rev, FS)

        if (i + 1) % 20 == 0 or i == len(df) - 1:
            el = time.time() - t0
            print("[%d/%d] utterances so far: %d (%d capped at %.0f s)  (%.1f s elapsed, %.2f s/RIR)"
                  % (i + 1, len(df), n_files, n_capped, MAX_S, el, el / (i + 1)), flush=True)

    rows = []
    for f in sorted(os.listdir(OUT_DIR)):
        with np.load(os.path.join(OUT_DIR, f)) as z:
            rows.append((f, float(z["t60"]), z["DR"].shape[1]))
    idx = pd.DataFrame(rows, columns=["filename", "T60_s", "n_frames"])
    idx.to_csv(INDEX_CSV, index=False)
    print("utterances: %d  total SDR frames: %d  -> %s"
          % (len(idx), idx["n_frames"].sum(), INDEX_CSV))


if __name__ == "__main__":
    main()
