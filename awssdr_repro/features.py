"""Paper-faithful AWSSDR spectral decay-rate extraction.

This is a cleaned port of the recovered ``compute_decayrates.py`` mechanics:
Hamming STFT -> log Mel magnitude -> per-band least-squares slopes.
Parameters are corrected to the paper values for from-scratch training.
"""

import librosa
import numpy as np

from . import config as cfg

_XT = (
    np.arange(1, cfg.FIT_LEN + 1, dtype=np.float64)
    * cfg.HOP_LENGTH
    / cfg.FS
).reshape(-1, 1)
_Q_SLOPE = np.linalg.pinv(np.concatenate([np.ones_like(_XT), _XT], axis=1))[1]
_MEL_BASIS = librosa.filters.mel(
    sr=cfg.FS,
    n_fft=cfg.N_FFT,
    n_mels=cfg.N_MELS,
    fmin=cfg.FMIN,
    fmax=cfg.FMAX,
)


def load_audio_16k(path):
    x, _ = librosa.load(path, sr=cfg.FS, mono=True)
    return x.astype(np.float32, copy=False)


def log_mel_magnitude(x):
    stft = librosa.stft(
        x,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        win_length=cfg.WIN_LENGTH,
        window="hamming",
        center=False,
    )
    return np.log(_MEL_BASIS @ np.abs(stft) + cfg.EPS)


def compute_decay_rates(x):
    if len(x) < cfg.MIN_SAMPLES:
        raise ValueError(
            f"waveform too short for SDR extraction: {len(x)} samples "
            f"< {cfg.MIN_SAMPLES}"
        )
    lmfe = log_mel_magnitude(x)
    windows = np.lib.stride_tricks.sliding_window_view(
        lmfe, cfg.FIT_LEN, axis=1
    )[:, :: cfg.FIT_HOP, :]
    return (windows @ _Q_SLOPE).astype(np.float32, copy=False)

