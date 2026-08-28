import json
from pathlib import Path

import numpy as np
import pandas as pd


def regression_metrics(df: pd.DataFrame) -> dict:
    err = df["target_t30"].to_numpy(dtype=float) - df["pred_t30"].to_numpy(dtype=float)
    if len(df) > 1:
        rho = float(np.corrcoef(df["target_t30"], df["pred_t30"])[0, 1])
    else:
        rho = float("nan")
    return {
        "bias": float(np.mean(err)),
        "mae": float(np.mean(np.abs(err))),
        "mse": float(np.mean(err ** 2)),
        "pearson_rho": rho,
        "n": int(len(df)),
    }


def grouped_metrics(df: pd.DataFrame, columns: list[str]) -> dict:
    out = {"overall": regression_metrics(df)}
    for col in columns:
        if col not in df.columns:
            continue
        out[col] = {
            str(key): regression_metrics(group)
            for key, group in df.groupby(col)
        }
    return out


def write_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True))

