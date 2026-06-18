"""
prepare_data.py
Generate the training/validation/test feature dataset for blind T60
estimation (Gamper & Tashev 2018, no-noise variant).

- Stratified RIR split (80/10/10 by 0.1 s T60 bins) + disjoint speakers
- Reverberant speech = clean EARS speech (resampled to 16 kHz) * RIR
- 4 s chunks, 0.5 s overlap, -20 dB RMS activity filter
- Features per features.py, stored float16 in one .npz per RIR
  (data/features/<split>/<rir>.npz with arrays X and y) -> resumable

Run:  python src/prepare_data.py
"""

import os
import sys
import glob
import time

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.signal import fftconvolve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FS, extract_features, chunk_signal  # noqa: E402

# --- Settings ---------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RIR_DIR = os.path.join(ROOT, "RIR_Data", "RIRS", "training_rirs_16khz_2")
LABELS_CSV = os.path.join(ROOT, "RIR_Data", "ground_truth_t60.csv")
EARS_DIR = os.path.join(ROOT, "Speech", "EARS")
DATA_DIR = os.path.join(ROOT, "data")
FEAT_DIR = os.path.join(DATA_DIR, "features")
DEBUG_DIR = os.path.join(DATA_DIR, "debug")

SEED = 42
UTTS_PER_RIR = 18          # utterances convolved with each RIR
SPEECH_PREFIXES = ("emo_", "rainbow_", "sentences_", "freeform_")

# Speaker split: disjoint across train/val (p009 does not exist in EARS-10)
SPEAKERS = {
    "train": ["p001", "p002", "p003", "p004", "p005", "p006", "p007", "p008"],
    "val":   ["p009","p010"],
}


def stratified_rir_split(df, rng):
    """80/20 train/val split stratified by 0.1 s T60 bins."""
    split = pd.Series("train", index=df.index)
    bins = (df["T60_s"] // 0.1).astype(int)
    for _, idx in df.groupby(bins).groups.items():
        idx = rng.permutation(np.array(idx))
        n = len(idx)
        n_val = max(1, round(0.2 * n)) if n >= 2 else 0
        split.loc[idx[:n_val]] = "val"
    return split


def index_speech():
    """{speaker: [wav paths]} for speech-only EARS categories."""
    out = {}
    for spk in sorted(os.listdir(EARS_DIR)):
        files = [f for f in glob.glob(os.path.join(EARS_DIR, spk, "*.wav"))
                 if os.path.basename(f).startswith(SPEECH_PREFIXES)]
        if files:
            out[spk] = sorted(files)
    return out


class SpeechCache:
    """Load+resample EARS files once, keep the most recent in RAM."""

    def __init__(self, max_items=400):
        self.cache = {}
        self.max_items = max_items

    def get(self, path):
        if path not in self.cache:
            if len(self.cache) >= self.max_items:
                self.cache.pop(next(iter(self.cache)))
            x, _ = librosa.load(path, sr=FS, mono=True)
            self.cache[path] = x.astype(np.float64)
        return self.cache[path]


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(FEAT_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    # --- Splits ---------------------------------------------------------
    df = pd.read_csv(LABELS_CSV)
    manifest_path = os.path.join(DATA_DIR, "split_manifest.csv")
    if os.path.exists(manifest_path):
        df = pd.read_csv(manifest_path)          # reuse: keeps runs resumable
        print(f"Reusing existing split manifest ({manifest_path})")
    else:
        df["split"] = stratified_rir_split(df, rng)
        df.to_csv(manifest_path, index=False)
    print(df["split"].value_counts().to_string())

    speech = index_speech()
    for split, spks in SPEAKERS.items():
        n = sum(len(speech.get(s, [])) for s in spks)
        print(f"{split}: speakers {spks} -> {n} speech files")

    cache = SpeechCache()
    t0 = time.time()
    total_chunks = 0
    index_rows = []
    wrote_debug = False

    for i, row in enumerate(df.itertuples()):
        split, rir_name, t60 = row.split, row.filename, row.T60_s
        out_dir = os.path.join(FEAT_DIR, split)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, rir_name.replace(".wav", ".npz"))

        if os.path.exists(out_path):             # resumable: skip done RIRs
            n_done = np.load(out_path)["y"].shape[0]
            total_chunks += n_done
            index_rows.append((rir_name, split, t60, n_done))
            continue

        rir, fs_r = sf.read(os.path.join(RIR_DIR, rir_name))
        assert fs_r == FS, f"{rir_name}: fs={fs_r}"
        if rir.ndim > 1:
            rir = rir[:, 0]
        rir = rir / (np.max(np.abs(rir)) + 1e-12)

        # Pick utterances from this split's speaker pool (seeded per RIR)
        pool = [f for s in SPEAKERS[split] for f in speech.get(s, [])]
        rng_rir = np.random.default_rng(SEED + i)
        picks = rng_rir.choice(len(pool), size=min(UTTS_PER_RIR, len(pool)),
                               replace=False)

        feats, labels = [], []
        for j in picks:
            x = cache.get(pool[j])
            rev = fftconvolve(x, rir)
            rev = rev / (np.max(np.abs(rev)) + 1e-12)

            if not wrote_debug:                  # one audible sanity check
                sf.write(os.path.join(DEBUG_DIR, "example_reverberant.wav"),
                         rev.astype(np.float32), FS)
                sf.write(os.path.join(DEBUG_DIR, "example_clean.wav"),
                         (x / (np.max(np.abs(x)) + 1e-12)).astype(np.float32), FS)
                print(f"debug audio written (RIR {rir_name}, T60={t60:.3f}s)")
                wrote_debug = True

            for c in chunk_signal(rev):
                feats.append(extract_features(c).astype(np.float16))
                labels.append(t60)

        X = np.stack(feats) if feats else np.zeros((0, 21, 1999), np.float16)
        y = np.array(labels, dtype=np.float32)
        np.savez_compressed(out_path, X=X, y=y)

        total_chunks += len(y)
        index_rows.append((rir_name, split, t60, len(y)))

        if (i + 1) % 10 == 0 or i == len(df) - 1:
            el = time.time() - t0
            print(f"[{i + 1:4d}/{len(df)}] {total_chunks} chunks | "
                  f"{el / 60:.1f} min elapsed | "
                  f"~{el / (i + 1) * (len(df) - i - 1) / 60:.0f} min left",
                  flush=True)

    idx = pd.DataFrame(index_rows,
                       columns=["filename", "split", "T60_s", "n_chunks"])
    idx.to_csv(os.path.join(DATA_DIR, "chunks_index.csv"), index=False)
    print("\nChunks per split:")
    print(idx.groupby("split")["n_chunks"].sum().to_string())
    print(f"\nTotal: {total_chunks} chunks -> {FEAT_DIR}")


if __name__ == "__main__":
    main()
