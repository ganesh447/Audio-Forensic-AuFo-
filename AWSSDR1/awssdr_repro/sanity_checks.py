import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .ears import build_ears_manifest
from .labels import load_ace_fb_t30_labels, load_train_t30_labels


def check_inputs():
    labels = load_train_t30_labels()
    wavs = {p.name for p in cfg.TRAIN_RIR_DIR.glob("*.wav")}
    clips = build_ears_manifest()
    print("RIR wavs", len(wavs))
    print("T30 labels", len(labels))
    print("missing labels", len(wavs - set(labels)))
    print("labels without wav", len(set(labels) - wavs))
    print("EARS clips", len(clips))
    print("EARS speakers", Counter(c.speaker for c in clips))
    print("EARS categories", Counter(c.category for c in clips))
    print("ACE dev FB T30", load_ace_fb_t30_labels("dev"))
    print("ACE eval FB T30", load_ace_fb_t30_labels("eval"))


def check_feature_dir(path: Path):
    manifest = pd.read_csv(path / "manifest.csv")
    assert (path / "dataset.pkl").exists()
    bad = []
    for row in manifest.itertuples(index=False):
        dr_path = path / "decayRates" / f"{row.sample_id}.npy"
        target_path = path / "T60" / f"{row.sample_id}.npy"
        if not dr_path.exists() or not target_path.exists():
            bad.append(row.sample_id)
            continue
        dr = np.load(dr_path)
        if dr.ndim != 2 or dr.shape[0] != cfg.N_MELS or dr.shape[1] <= 0:
            bad.append(row.sample_id)
        if not np.isfinite(dr).all():
            bad.append(row.sample_id)
    print(path, "rows", len(manifest), "bad", len(bad))
    if bad[:10]:
        print("bad examples", bad[:10])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=None)
    args = parser.parse_args()
    check_inputs()
    if args.features:
        check_feature_dir(args.features)


if __name__ == "__main__":
    main()
