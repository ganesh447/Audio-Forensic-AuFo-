import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.Model_structure import RT_est

from . import config as cfg
from .dataset_io import load_dataset_ids
from .metrics import grouped_metrics, write_metrics


def load_model(checkpoint_dir: Path, device):
    model = RT_est(num_channels=cfg.N_MELS, fc_dim=cfg.TRAINING["fc_dim"]).to(device)
    weights = Path(checkpoint_dir) / "latest_weights.pyt"
    model.load_state_dict(torch.load(weights, map_location=device), strict=False)
    model.eval()
    return model


def evaluate(feature_dir: Path, checkpoint_dir: Path, out_csv: Path, metrics_json: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_dir, device)
    ids = [x[0] for x in load_dataset_ids(feature_dir)]
    manifest_path = feature_dir / "manifest.csv"
    manifest = pd.read_csv(manifest_path).set_index("sample_id") if manifest_path.exists() else None
    rows = []
    with torch.no_grad():
        for sample_id in ids:
            dr = np.load(feature_dir / "decayRates" / f"{sample_id}.npy")
            target = float(np.load(feature_dir / "T60" / f"{sample_id}.npy"))
            x = torch.from_numpy(dr.T).view(1, -1, cfg.N_MELS).to(device, torch.float32)
            pred, *_ = model(x)
            pred_t30 = float(pred.detach().cpu().view(-1)[0])
            row = {
                "sample_id": sample_id,
                "target_t30": target,
                "pred_t30": pred_t30,
                "err_s": target - pred_t30,
            }
            if manifest is not None and sample_id in manifest.index:
                row.update(manifest.loc[sample_id].to_dict())
            rows.append(row)
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    metrics = grouped_metrics(
        df,
        ["room", "room_config", "noise_type", "snr_db", "talker", "utterance", "gender"],
    )
    write_metrics(metrics_json, metrics)
    print(metrics["overall"])
    print(f"wrote {out_csv}")
    print(f"wrote {metrics_json}")
    return df, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=cfg.ACE_EVAL_FEATURE_DIR)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=cfg.REPO_ROOT / "checkpoints" / f"{cfg.CHECKPOINT_ID}.dnn",
    )
    parser.add_argument("--out", type=Path, default=cfg.REPORTS_DIR / "eval_predictions.csv")
    parser.add_argument("--metrics", type=Path, default=cfg.REPORTS_DIR / "metrics.json")
    args = parser.parse_args()
    evaluate(args.features, args.checkpoint, args.out, args.metrics)


if __name__ == "__main__":
    main()

