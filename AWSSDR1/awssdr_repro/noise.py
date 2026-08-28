import glob
import os
import re
from pathlib import Path

import librosa
import numpy as np
import scipy.ndimage
import scipy.signal
from scipy.signal import fftconvolve

from . import config as cfg
from .ears import SpeechClip
from .features import load_audio_16k

_speech_cache: dict[Path, np.ndarray] = {}


def load_rir(path: Path) -> np.ndarray:
    x = load_audio_16k(path)
    peak = np.max(np.abs(x))
    if peak == 0:
        raise ValueError(f"silent RIR: {path}")
    return x / (peak + 1e-12)


def load_noise_pool() -> dict[str, list[np.ndarray]]:
    pool = {"Ambient": [], "Fan": []}
    for room in cfg.NOISE_ROOMS:
        for pos in ("1", "2"):
            for kind in pool:
                paths = glob.glob(
                    str(cfg.ACE_RIRN_DIR / room / pos / f"*_Noise_{kind}.wav")
                )
                if len(paths) != 1:
                    raise FileNotFoundError((room, pos, kind, paths))
                pool[kind].append(load_audio_16k(paths[0]))
    return pool


def load_transient_pool() -> list[tuple[np.ndarray, float, str]]:
    pool = []
    for cls in cfg.TRANSIENT_CLASSES:
        pattern = re.compile(rf"^{re.escape(cls)}_\d+_{re.escape(cfg.TRANSIENT_MIC)}\.wav$")
        paths = sorted(
            p for p in cfg.AID_DIR.glob("*.wav")
            if pattern.match(os.path.basename(p))
        )
        if not paths:
            raise FileNotFoundError(f"no AID files for {cls} on {cfg.TRANSIENT_MIC}")
        for path in paths:
            x = load_audio_16k(path)
            x, _ = librosa.effects.trim(x, top_db=30)
            active = np.abs(x) > 0.05 * (np.max(np.abs(x)) + 1e-12)
            if active.any():
                pool.append((x, float(np.sqrt(np.mean(x[active] ** 2))), cls))
    if not pool:
        raise FileNotFoundError("no usable AID transient clips")
    return pool


def load_speech_cached(clip: SpeechClip) -> np.ndarray:
    if clip.path not in _speech_cache:
        if len(_speech_cache) >= 1000:
            _speech_cache.pop(next(iter(_speech_cache)))
        _speech_cache[clip.path] = load_audio_16k(clip.path)
    return _speech_cache[clip.path]


def add_transients(rng, bed: np.ndarray, transient_pool) -> np.ndarray:
    bed = bed.copy()
    bed_rms = np.sqrt(np.mean(bed ** 2)) + 1e-12
    n_events = rng.integers(cfg.N_TRANSIENTS[0], cfg.N_TRANSIENTS[1] + 1)
    for _ in range(n_events):
        transient, transient_rms, _ = transient_pool[rng.integers(len(transient_pool))]
        gain = bed_rms / (transient_rms + 1e-12)
        gain *= 10.0 ** (rng.uniform(*cfg.TRANSIENT_GAIN_DB) / 20.0)
        offset = rng.integers(max(len(bed) - len(transient), 0) + 1)
        segment = transient[: len(bed) - offset]
        bed[offset : offset + len(segment)] += gain * segment
    return bed


def shaped_wgn(rng, noise_pool, kind: str, n: int) -> np.ndarray:
    rec = noise_pool[kind][rng.integers(len(noise_pool[kind]))]
    seg_n = min(int(cfg.NOISE_SEG_S * cfg.FS), len(rec))
    start = rng.integers(len(rec) - seg_n + 1)
    seg = rec[start : start + seg_n]
    _, _, z = scipy.signal.stft(seg, cfg.FS, nperseg=cfg.STFT_NPERSEG)
    env = np.sqrt((np.abs(z) ** 2).mean(axis=1))

    wgn = rng.standard_normal(n)
    _, _, w = scipy.signal.stft(wgn, cfg.FS, nperseg=cfg.STFT_NPERSEG)
    _, out = scipy.signal.istft(w * env[:, None], cfg.FS, nperseg=cfg.STFT_NPERSEG)
    if len(out) < n:
        out = np.pad(out, (0, n - len(out)))
    return out[:n]


def make_noise(
    rng,
    noise_pool,
    kind: str,
    rir: np.ndarray,
    n: int,
    speech_pool: list[SpeechClip],
    target_clip: SpeechClip,
    transient_pool,
) -> np.ndarray:
    if kind in ("Ambient", "Fan"):
        dry_bed = shaped_wgn(rng, noise_pool, kind, n)
        dry_bed = add_transients(rng, dry_bed, transient_pool)
        return fftconvolve(dry_bed, rir)[:n]

    if kind != "Babble":
        raise ValueError(f"unknown noise kind: {kind}")
    n_talkers = rng.integers(cfg.N_BABBLE_TALKERS[0], cfg.N_BABBLE_TALKERS[1] + 1)
    babble = np.zeros(n, dtype=np.float64)
    for _ in range(n_talkers):
        clip = speech_pool[rng.integers(len(speech_pool))]
        while clip.path == target_clip.path:
            clip = speech_pool[rng.integers(len(speech_pool))]
        rev = fftconvolve(load_speech_cached(clip), rir)
        reps = int(np.ceil(n / len(rev))) + 1
        offset = rng.integers(len(rev))
        babble += np.tile(rev, reps)[offset : offset + n]

    amb_bed = add_transients(rng, shaped_wgn(rng, noise_pool, "Ambient", n), transient_pool)
    amb = fftconvolve(amb_bed, rir)[:n]
    amb *= (np.sqrt(np.mean(babble ** 2)) / (np.sqrt(np.mean(amb ** 2)) + 1e-12))
    amb *= 10.0 ** (cfg.BABBLE_AMBIENT_DB / 20.0)
    return babble + amb


def active_speech_level(x: np.ndarray) -> float:
    sq = float(np.sum(x ** 2))
    peak = float(np.max(np.abs(x)))
    if peak == 0.0 or sq == 0.0:
        return 0.0
    g = np.exp(-1.0 / (cfg.FS * 0.03))
    p = scipy.signal.lfilter([1 - g], [1, -g], np.abs(x))
    q = scipy.signal.lfilter([1 - g], [1, -g], p)
    hang = int(np.ceil(0.2 * cfg.FS))
    qmax = scipy.ndimage.maximum_filter1d(
        q, size=hang + 1, origin=-(hang // 2), mode="constant", cval=0.0
    )
    c = peak * 2.0 ** -np.arange(15, 0, -1)
    a = len(x) - np.searchsorted(np.sort(qmax), c)
    valid = a > 0
    c, a = c[valid], a[valid]
    A = 10 * np.log10(sq / a)
    C = 20 * np.log10(c)
    d = A - C
    margin_db = 15.9
    if d[0] <= margin_db:
        return float(sq / a[0])
    below = np.nonzero(d <= margin_db)[0]
    if len(below) == 0:
        return float(sq / a[-1])
    j = below[0]
    frac = (d[j - 1] - margin_db) / (d[j - 1] - d[j])
    return float(10.0 ** ((A[j - 1] + frac * (A[j] - A[j - 1])) / 10.0))


def mix_at_snr(reverberant_speech: np.ndarray, noise: np.ndarray, snr_db: int) -> np.ndarray:
    speech_power = active_speech_level(reverberant_speech)
    noise_power = float(np.mean(noise ** 2))
    if speech_power <= 0 or noise_power <= 0:
        raise ValueError("cannot mix silent speech/noise")
    gain = np.sqrt(speech_power / (noise_power * 10.0 ** (snr_db / 10.0)))
    mix = reverberant_speech + gain * noise
    return (mix / (np.max(np.abs(mix)) + 1e-12)).astype(np.float32, copy=False)

