"""
T60 prediction on a directory of wav files — no ground truth needed.
Usage: python predict.py
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from math import gcd

import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
from scipy.signal import resample_poly, bilinear, lfilter, freqz
from gammatone.filters import erb_space, make_erb_filters, erb_filterbank


# ── A-weighting (matches training pipeline exactly, including original bug) ────

def _build_a_weighting(fs: int):
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    p = 2 * np.pi
    num = [(p * f4) ** 2, 0, 0, 0, 0]
    den = np.polymul(
        np.polymul([1, 4*p*f4, (p*f4)**2], [1, 4*p*f1, (p*f1)**2]),
        np.polymul([1, p*f3],               [1, p*f2]),
    )
    b, a = bilinear(num, den, fs)
    _, h = freqz(b, a, worN=[1000], fs=fs)
    b /= abs(h[0])
    return b, a

_AW_CACHE: dict = {}

def _apply_a_weighting(signal: np.ndarray, fs: int) -> np.ndarray:
    if fs not in _AW_CACHE:
        _AW_CACHE[fs] = _build_a_weighting(fs)
    b, a = _AW_CACHE[fs]
    return lfilter(b, a, signal).astype(np.float32)


# ── Model (must match training) ────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel,
                      stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class CNNT60(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 5, (1, 10), (1, 2)),
            ConvBlock(5, 5, (1, 10), (1, 3)),
            ConvBlock(5, 5, (1, 11), (1, 3)),
            ConvBlock(5, 5, (1, 11), (1, 2)),
            ConvBlock(5, 5, (3,  8), (2, 2)),
            ConvBlock(5, 5, (4,  7), (2, 1)),
        )
        self.dropout = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten()
        self.dense   = nn.Linear(5 * 4 * 15, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x.squeeze(1)


# ── Predictor ──────────────────────────────────────────────────────────────────

class T60Predictor:

    TARGET_SR      = 16_000
    WINDOW_SAMPLES = 64_000
    HOP_SAMPLES    = 8_000
    SILENCE_DB     = 20.0
    N_BANDS        = 21
    FREQ_LOW       = 400.0
    FREQ_HIGH      = 6_000.0
    FRAME_SIZE     = 64
    FRAME_HOP      = 32

    def __init__(self, checkpoint: Path, global_mean: Path, global_std: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CNNT60().to(self.device)
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True))
        self.model.eval()

        self.global_mean = np.load(str(global_mean))
        self.global_std  = np.load(str(global_std))

        print(f"Model loaded  : {checkpoint}")
        print(f"Device        : {self.device}\n")

    def _load_wav(self, path: Path) -> np.ndarray:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        if sr != self.TARGET_SR:
            g    = gcd(self.TARGET_SR, sr)
            data = resample_poly(data, self.TARGET_SR // g,
                                 sr // g).astype(np.float32)
        return data

    def _rms_db(self, x: np.ndarray) -> float:
        return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-10)

    def _extract_features(self, chunk: np.ndarray) -> np.ndarray:
        chunk    = _apply_a_weighting(chunk, self.TARGET_SR)
        cfs      = erb_space(self.FREQ_LOW, self.FREQ_HIGH, self.N_BANDS)
        fcoefs   = make_erb_filters(self.TARGET_SR, cfs)
        filtered = erb_filterbank(chunk.astype(float), fcoefs).astype(np.float32)
        n_frames = (filtered.shape[1] - self.FRAME_SIZE) // self.FRAME_HOP + 1
        feat     = np.empty((self.N_BANDS, n_frames), dtype=np.float32)
        for i in range(n_frames):
            s          = i * self.FRAME_HOP
            frame      = filtered[:, s : s + self.FRAME_SIZE]
            feat[:, i] = np.log(np.sum(frame ** 2, axis=1) + 1e-10)
        feat = feat - np.median(feat, axis=1, keepdims=True)
        feat = (feat - self.global_mean) / (self.global_std + 1e-8)
        return feat.astype(np.float32)

    def predict_file(self, wav_path: Path) -> float:
        audio    = self._load_wav(wav_path)
        clip_rms = self._rms_db(audio)
        chunks   = []
        start    = 0
        while start + self.WINDOW_SAMPLES <= len(audio):
            chunk = audio[start : start + self.WINDOW_SAMPLES]
            if self._rms_db(chunk) >= clip_rms - self.SILENCE_DB:
                peak = np.max(np.abs(chunk))
                if peak > 1e-9:
                    chunk = chunk / peak
                chunks.append(self._extract_features(chunk))
            start += self.HOP_SAMPLES

        if not chunks:
            return float("nan")

        preds = []
        with torch.no_grad():
            for feat in chunks:
                x    = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0)
                pred = self.model(x.to(self.device)).item()
                preds.append(pred)
        return float(np.mean(preds))

    def predict_directory(self, wav_dir: Path):
        wav_files = sorted(wav_dir.glob("*.wav"))
        if not wav_files:
            print(f"No .wav files found in {wav_dir}")
            return {}

        print(f"Found {len(wav_files)} wav files in {wav_dir}\n")
        print(f"{'File':<50}  T60 estimate")
        print("-" * 65)

        results = {}
        for wav_path in wav_files:
            pred = self.predict_file(wav_path)
            if np.isnan(pred):
                print(f"{wav_path.name:<50}  [skipped — too short or silent]")
            else:
                print(f"{wav_path.name:<50}  {pred:.3f} s")
                results[wav_path.name] = pred

        if results:
            values = list(results.values())
            print(f"\n── Summary ───────────────────────────────────")
            print(f"  Files predicted : {len(results)}")
            print(f"  Min T60         : {min(values):.3f} s")
            print(f"  Max T60         : {max(values):.3f} s")
            print(f"  Mean T60        : {np.mean(values):.3f} s")

        return results


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    SCRIPT_DIR  = Path(__file__).resolve().parent
    PROJECT_DIR = SCRIPT_DIR.parent

    CHECKPOINT  = SCRIPT_DIR / "checkpoints" / "best_modelv1.pt"
    GLOBAL_MEAN = SCRIPT_DIR / "global_mean.npy"
    GLOBAL_STD  = SCRIPT_DIR / "global_std.npy"

    # ── Point this at your folder of wav files ─────────────────────────────────
    WAV_DIR     = PROJECT_DIR / r"Datasets" / r"test_wavs" / r"convolved_sim_largeroom_Room024_Room024-00001_T60_0.84"
    predictor = T60Predictor(
        checkpoint  = CHECKPOINT,
        global_mean = GLOBAL_MEAN,
        global_std  = GLOBAL_STD,
    )

    results = predictor.predict_directory(WAV_DIR)
