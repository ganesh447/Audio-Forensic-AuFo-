import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def safe_stem(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text.strip())
    return text.strip("-")


def prepare_feature_dir(out_dir: Path) -> None:
    (out_dir / "decayRates").mkdir(parents=True, exist_ok=True)
    (out_dir / "T60").mkdir(parents=True, exist_ok=True)


def save_sample(out_dir: Path, sample_id: str, dr: np.ndarray, target_t30: float) -> None:
    np.save(out_dir / "decayRates" / f"{sample_id}.npy", dr.astype(np.float32, copy=False))
    np.save(out_dir / "T60" / f"{sample_id}.npy", np.asarray(target_t30, dtype=np.float32))


def write_index(out_dir: Path, rows: list[dict]) -> None:
    dataset = [(r["sample_id"], int(r["n_decay_frames"])) for r in rows]
    with open(out_dir / "dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)
    pd.DataFrame(rows).to_csv(out_dir / "manifest.csv", index=False)


class DecayRateDataset(Dataset):
    def __init__(self, feature_dir: Path, ids: list[str]):
        self.feature_dir = Path(feature_dir)
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample_id = self.ids[index]
        dr = np.load(self.feature_dir / "decayRates" / f"{sample_id}.npy")
        target = np.load(self.feature_dir / "T60" / f"{sample_id}.npy")
        return dr, target


def collate_decay_rates(samples):
    dr = [torch.from_numpy(sample[0].T) for sample in samples]
    target = [torch.as_tensor(sample[1], dtype=torch.float32) for sample in samples]
    padded = torch.nn.utils.rnn.pad_sequence(dr, batch_first=True, padding_value=0.0)
    return padded.contiguous(), torch.stack(target).contiguous()


def load_dataset_ids(feature_dir: Path) -> list[tuple[str, int]]:
    with open(Path(feature_dir) / "dataset.pkl", "rb") as f:
        return pickle.load(f)


def make_loader(
    feature_dir: Path,
    batch_size: int,
    shuffle_batches: bool,
    num_workers: int = 8,
) -> DataLoader:
    dataset = load_dataset_ids(feature_dir)
    ids = [x[0] for x in dataset]
    lens = [int(x[1]) for x in dataset]
    lens, ids = zip(*sorted(zip(lens, ids), reverse=True))
    torch_dataset = DecayRateDataset(feature_dir, list(ids))

    indices = list(range(len(torch_dataset)))
    batches = [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]
    if shuffle_batches:
        g = torch.Generator()
        g.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        order = torch.randperm(len(batches), generator=g).tolist()
        batches = [batches[i] for i in order]

    return DataLoader(
        torch_dataset,
        batch_sampler=batches,
        collate_fn=collate_decay_rates,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
    )
