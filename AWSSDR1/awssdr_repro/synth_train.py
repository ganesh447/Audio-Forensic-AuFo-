import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

from . import config as cfg
from .dataset_io import prepare_feature_dir, safe_stem, save_sample, write_index
from .ears import build_ears_manifest, group_by_speaker, select_balanced_clips
from .features import compute_decay_rates, load_audio_16k
from .labels import load_train_t30_labels
from .noise import load_noise_pool, load_rir, load_transient_pool, make_noise, mix_at_snr


def build(out_dir: Path, limit_rirs: int = 0, utts_per_rir: int = cfg.K_UTT_PER_RIR):
    labels = load_train_t30_labels()
    rir_paths = sorted(p for p in cfg.TRAIN_RIR_DIR.glob("*.wav") if p.name in labels)
    if limit_rirs:
        rir_paths = rir_paths[:limit_rirs]
    if not rir_paths:
        raise FileNotFoundError("no labelled RIR wavs found")

    speech_pool = build_ears_manifest()
    clips_by_speaker = group_by_speaker(speech_pool)
    noise_pool = load_noise_pool()
    transient_pool = load_transient_pool()
    prepare_feature_dir(out_dir)

    rows = []
    start = time.time()
    for rir_index, rir_path in enumerate(rir_paths):
        rir = load_rir(rir_path)
        rng = np.random.default_rng([cfg.SEED, rir_index])
        target_clips = select_balanced_clips(clips_by_speaker, rng, utts_per_rir)
        target_t30 = labels[rir_path.name]
        rir_id = safe_stem(rir_path.stem)

        for utt_index, clip in enumerate(target_clips):
            condition = utt_index % (len(cfg.NOISE_TYPES) * len(cfg.SNRS_DB))
            noise_type = cfg.NOISE_TYPES[condition // len(cfg.SNRS_DB)]
            snr_db = cfg.SNRS_DB[condition % len(cfg.SNRS_DB)]
            sample_id = f"train__rir_{rir_id}__u{utt_index:02d}__{noise_type}__{snr_db}dB"

            speech = load_audio_16k(clip.path)
            reverberant = fftconvolve(speech, rir)
            if len(reverberant) < cfg.MIN_SAMPLES:
                continue
            noise = make_noise(
                rng,
                noise_pool,
                noise_type,
                rir,
                len(reverberant),
                speech_pool,
                clip,
                transient_pool,
            )
            mixture = mix_at_snr(reverberant, noise, snr_db)
            dr = compute_decay_rates(mixture)
            save_sample(out_dir, sample_id, dr, target_t30)
            rows.append(
                {
                    "sample_id": sample_id,
                    "split": "train",
                    "rir_filename": rir_path.name,
                    "target_t30": target_t30,
                    "speech_path": str(clip.path),
                    "speech_speaker": clip.speaker,
                    "speech_category": clip.category,
                    "noise_type": noise_type,
                    "snr_db": snr_db,
                    "n_samples": len(mixture),
                    "n_decay_frames": dr.shape[1],
                }
            )

        if (rir_index + 1) % 10 == 0 or rir_index == len(rir_paths) - 1:
            elapsed = time.time() - start
            print(
                f"[{rir_index + 1}/{len(rir_paths)}] samples={len(rows)} "
                f"elapsed={elapsed:.1f}s"
            )

    write_index(out_dir, rows)
    print(f"wrote {len(rows)} samples to {out_dir}")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=cfg.TRAIN_FEATURE_DIR)
    parser.add_argument("--limit-rirs", type=int, default=0)
    parser.add_argument("--utts-per-rir", type=int, default=cfg.K_UTT_PER_RIR)
    args = parser.parse_args()
    build(args.out, args.limit_rirs, args.utts_per_rir)


if __name__ == "__main__":
    main()

