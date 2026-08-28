import argparse
from pathlib import Path

from . import config as cfg
from .dataset_io import prepare_feature_dir, safe_stem, save_sample, write_index
from .features import compute_decay_rates, load_audio_16k
from .labels import load_ace_fb_t30_labels, parse_ace_filename


def split_paths(split: str):
    if split == "dev":
        return cfg.ACE_DEV_DIR, cfg.ACE_DEV_FEATURE_DIR
    if split == "eval":
        return cfg.ACE_EVAL_DIR, cfg.ACE_EVAL_FEATURE_DIR
    raise ValueError("split must be 'dev' or 'eval'")


def build(split: str, out_dir: Path | None = None, limit: int = 0):
    wav_dir, default_out = split_paths(split)
    out_dir = out_dir or default_out
    labels = load_ace_fb_t30_labels(split)
    wavs = sorted(wav_dir.glob("*.wav"))
    if limit:
        wavs = wavs[:limit]
    prepare_feature_dir(out_dir)
    rows = []
    for idx, wav in enumerate(wavs):
        meta = parse_ace_filename(wav.name)
        target = labels[meta["room_config"]]
        x = load_audio_16k(wav)
        dr = compute_decay_rates(x)
        sample_id = safe_stem(wav.stem)
        save_sample(out_dir, sample_id, dr, target)
        rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "wav_path": str(wav),
                "target_t30": target,
                "n_samples": len(x),
                "n_decay_frames": dr.shape[1],
                **meta,
            }
        )
        if (idx + 1) % 250 == 0 or idx == len(wavs) - 1:
            print(f"[{idx + 1}/{len(wavs)}] {split}")
    write_index(out_dir, rows)
    print(f"wrote {len(rows)} {split} samples to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "eval"), required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    build(args.split, args.out, args.limit)


if __name__ == "__main__":
    main()

