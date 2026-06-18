"""
prepare_ace_data.py
Build the ACE evaluation feature set for blind T60 estimation.

Unlike the recorded (already reverberant+noisy) ACE wavs, this script
builds the test set the same way prepare_data.py builds train/val:
convolve clean speech with RIRs ourselves.

- RIRs:   Eval_data/ACE eval/ACE_Corpus_RIRN_Single/Single/<Room>/<Pos>/*_RIR.wav
- Speech: Eval_data/ACE eval/Speech/*.wav (clean utterances)
- Labels: Eval_data/ground_truth_t60.csv (T60 per RIR, T30 fullband via
  ITA-Toolbox)

Both RIRs and speech are 48 kHz; resampled to 16 kHz before convolution.
Features per features.py, stored float16 in one .npz per RIR
(data/features/test/<Room>_<Pos>.npz with arrays X and y).

Run:  python src/prepare_ace_data.py
"""

import os
import sys
import glob

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.signal import fftconvolve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FS, extract_features, chunk_signal  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACE_DIR = os.path.join(ROOT, "Eval_data", "ACE eval")
RIR_DIR = os.path.join(ACE_DIR, "ACE_Corpus_RIRN_Single", "Single")
SPEECH_DIR = os.path.join(ACE_DIR, "Speech")
LABELS_CSV = os.path.join(ACE_DIR, "ground_truth_t60.csv")
OUT_DIR = os.path.join(ROOT, "data", "features", "test")


def load_labels():
    """{(Room, Position): T60_s} keyed by the two path components just
    above the RIR filename (the CSV's stored absolute paths are stale,
    so only the trailing Room/Position/filename structure is used)."""
    df = pd.read_csv(LABELS_CSV)
    labels = {}
    for row in df.itertuples():
        parts = row.filename.replace("\\", "/").split("/")
        room, pos = parts[-3], parts[-2]
        labels[(room, pos)] = row.T60_s
    return labels


def index_rirs():
    paths = sorted(glob.glob(os.path.join(RIR_DIR, "**", "*_RIR.wav"),
                              recursive=True))
    rirs = []
    for p in paths:
        rel = os.path.relpath(p, RIR_DIR).replace("\\", "/").split("/")
        room, pos = rel[0], rel[1]
        rirs.append((room, pos, p))
    return rirs


def index_speech():
    return sorted(glob.glob(os.path.join(SPEECH_DIR, "*.wav")))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    labels = load_labels()
    rirs = index_rirs()
    speech_files = index_speech()
    print(f"{len(rirs)} ACE RIRs, {len(speech_files)} clean speech files")

    speech_cache = {}

    def get_speech(path):
        if path not in speech_cache:
            x, _ = librosa.load(path, sr=FS, mono=True)
            speech_cache[path] = x.astype(np.float64)
        return speech_cache[path]

    for i, (room, pos, rir_path) in enumerate(rirs):
        t60 = labels[(room, pos)]
        out_path = os.path.join(OUT_DIR, f"{room}_{pos}.npz")
        if os.path.exists(out_path):              # resumable
            continue

        rir, fs_r = sf.read(rir_path)
        if rir.ndim > 1:
            rir = rir[:, 0]
        if fs_r != FS:
            rir = librosa.resample(rir.astype(np.float64), orig_sr=fs_r,
                                    target_sr=FS)
        rir = rir / (np.max(np.abs(rir)) + 1e-12)

        feats, lbls = [], []
        for sp_path in speech_files:
            x = get_speech(sp_path)
            rev = fftconvolve(x, rir)
            rev = rev / (np.max(np.abs(rev)) + 1e-12)

            for c in chunk_signal(rev):
                feats.append(extract_features(c).astype(np.float16))
                lbls.append(t60)

        X = np.stack(feats) if feats else np.zeros((0, 21, 1999), np.float16)
        y = np.array(lbls, dtype=np.float32)
        np.savez_compressed(out_path, X=X, y=y)

        print(f"[{i + 1:2d}/{len(rirs)}] {room}/{pos} "
              f"(T60={t60:.3f}s) -> {len(y)} chunks")

    print(f"\nDone -> {OUT_DIR}")


if __name__ == "__main__":
    main()
