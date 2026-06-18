"""
features.py
Feature extraction for blind T60 estimation, following
Gamper & Tashev, "Blind reverberation time estimation using a
convolutional neural network", IWAENC 2018, Section 3.2.

Pipeline per 4-second chunk (16 kHz):
  1. Level normalisation using an A-weighting filter
  2. Gammatone ERB filterbank, 21 bands, 400 Hz - 6 kHz
  3. Per band: log of energy summed in frames of 64 samples, hop 32
  4. Spectral whitening: subtract per-band median
  5. Normalise to ~zero mean, unit std
Result: float32 feature matrix of shape (21, 1999) per chunk.
"""

import numpy as np
from scipy.signal import bilinear, lfilter
from gammatone import filters as gtf

FS = 16000
N_BANDS = 21
F_LOW = 400.0
F_HIGH = 6000.0
FRAME_LEN = 64
FRAME_HOP = 32
CHUNK_SEC = 4.0
CHUNK_SAMPLES = int(CHUNK_SEC * FS)          # 64000
N_FRAMES = (CHUNK_SAMPLES - FRAME_LEN) // FRAME_HOP + 1   # 1999
EPS = 1e-12


def _a_weighting_coeffs(fs):
    """Digital A-weighting filter coefficients (IEC 61672 analog
    prototype mapped with the bilinear transform)."""
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    a1000 = 1.9997  # gain normalisation at 1 kHz (dB)

    num = [(2 * np.pi * f4) ** 2 * 10 ** (a1000 / 20.0), 0, 0, 0, 0]
    den = np.polymul([1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2],
                     [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2])
    den = np.polymul(np.polymul(den, [1, 2 * np.pi * f3]),
                     [1, 2 * np.pi * f2])
    return bilinear(num, den, fs)


_A_B, _A_A = _a_weighting_coeffs(FS)

# Gammatone filterbank coefficients (21 ERB-spaced bands, 400-6000 Hz).
# erb_space returns descending frequencies; flip so row 0 = lowest band.
_CENTRE_FREQS = np.sort(gtf.erb_space(F_LOW, F_HIGH, N_BANDS))
_GT_COEFS = gtf.make_erb_filters(FS, _CENTRE_FREQS)


def a_weighted_rms(x):
    """RMS of the A-weighted signal."""
    y = lfilter(_A_B, _A_A, x)
    return np.sqrt(np.mean(y ** 2) + EPS)


def extract_features(chunk):
    """Compute the (21, 1999) feature matrix for one 4-s chunk @16 kHz.

    chunk : 1-D float array, exactly CHUNK_SAMPLES long (zero-pad shorter).
    """
    if len(chunk) < CHUNK_SAMPLES:
        chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
    else:
        chunk = chunk[:CHUNK_SAMPLES]
    chunk = chunk.astype(np.float64)

    # 1. A-weighted level normalisation
    chunk = chunk / a_weighted_rms(chunk)

    # 2. Gammatone filterbank -> (21, 64000)
    bands = gtf.erb_filterbank(chunk, _GT_COEFS)

    # 3. Framed log energies -> (21, 1999)
    # Frame with a strided view: (21, N_FRAMES, FRAME_LEN)
    sq = bands ** 2
    idx = np.arange(N_FRAMES) * FRAME_HOP
    frames = np.lib.stride_tricks.sliding_window_view(
        sq, FRAME_LEN, axis=1)[:, idx, :]
    feat = np.log(frames.sum(axis=2) + EPS)

    # 4. Spectral whitening: subtract per-band median
    feat -= np.median(feat, axis=1, keepdims=True)

    # 5. Standardise to ~zero mean, unit std
    feat = (feat - feat.mean()) / (feat.std() + EPS)

    return feat.astype(np.float32)


def chunk_signal(x, full_rms_db_drop=20.0, hop_sec=3.5):
    """Split a signal into 4-s chunks with 0.5 s overlap (3.5 s hop)
    and discard chunks whose RMS is more than `full_rms_db_drop` dB
    below the RMS of the entire signal. Yields 1-D arrays of
    CHUNK_SAMPLES samples (last chunk zero-padded)."""
    full_rms = np.sqrt(np.mean(x ** 2) + EPS)
    thresh = full_rms * 10 ** (-full_rms_db_drop / 20.0)

    hop = int(hop_sec * FS)
    n = len(x)
    starts = list(range(0, max(n - CHUNK_SAMPLES, 0) + 1, hop))
    if not starts:                       # shorter than 4 s: single padded chunk
        starts = [0]

    for s in starts:
        c = x[s:s + CHUNK_SAMPLES]
        rms = np.sqrt(np.mean(c ** 2) + EPS)
        if rms >= thresh:
            if len(c) < CHUNK_SAMPLES:
                c = np.pad(c, (0, CHUNK_SAMPLES - len(c)))
            yield c


if __name__ == "__main__":
    # Self-test on white noise
    rng = np.random.default_rng(0)
    x = rng.standard_normal(CHUNK_SAMPLES)
    f = extract_features(x)
    print(f"feature shape: {f.shape} (expect (21, {N_FRAMES}))")
    print(f"mean: {f.mean():.4f} (expect ~0) | std: {f.std():.4f} (expect ~1)")
    print(f"centre freqs: {_CENTRE_FREQS.round(1)}")
    chunks = list(chunk_signal(rng.standard_normal(FS * 10)))
    print(f"10 s noise -> {len(chunks)} chunks (expect 2)")
