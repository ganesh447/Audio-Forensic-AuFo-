from pathlib import Path

import pandas as pd

from . import config as cfg


def load_train_t30_labels(csv_path: Path = cfg.TRAIN_LABEL_CSV) -> dict[str, float]:
    df = pd.read_csv(csv_path)
    required = {"filename", "t30"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
    return {str(r.filename): float(r.t30) for r in df.itertuples(index=False)}


def parse_ace_filename(path_or_name):
    stem = Path(path_or_name).stem
    parts = stem.split("_")
    if len(parts) < 7 or parts[0] != "Single":
        raise ValueError(f"unexpected ACE filename: {path_or_name}")
    snr_db = int(parts[-1].replace("dB", ""))
    noise_type = parts[-2]
    utterance = parts[-3]
    talker = parts[-4]
    config = parts[-5]
    room = "_".join(parts[1:-5])
    gender = "Female" if talker.startswith("F") else "Male"
    return {
        "room": room,
        "config": config,
        "room_config": f"{room}_{config}",
        "talker": talker,
        "gender": gender,
        "utterance": utterance,
        "noise_type": noise_type,
        "snr_db": snr_db,
    }


def load_ace_fb_t30_labels(split: str) -> dict[str, float]:
    if split == "dev":
        csv_path = cfg.ACE_DEV_CSV
    elif split == "eval":
        csv_path = cfg.ACE_EVAL_CSV
    else:
        raise ValueError("split must be 'dev' or 'eval'")
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [c.strip().rstrip(":") for c in df.columns]
    target_col = "FB T30 ISO Mean (Ch)"
    required = {"Room", "Room Config", target_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
    grouped = df.groupby(["Room", "Room Config"])[target_col].first()
    return {
        f"{room}_{config}": float(value)
        for (room, config), value in grouped.items()
    }

