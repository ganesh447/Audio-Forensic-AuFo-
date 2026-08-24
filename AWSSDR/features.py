"""Feature extraction: waveform -> spectral decay rate (SDR) sequence (40, N).

Pipeline (Kim & Kim 2022, Sec. III-B; mechanics match the reference compute_decayrates.py
recovered by decompiling Attention_Pooling/RTest-main/__pycache__/compute_decayrates.*.pyc):
LMFE = natural log of a 40-band mel spectrogram of |STFT| (Hamming 16 ms / 8 ms shift,
center=False, 400-6000 Hz), then per band a linear least-squares line is fit to every
window of FIT_LEN=40 LMFE frames, hop FIT_HOP=2 frames; the slope is the SDR.

Two reference-code quirks are resolved toward the paper text:
  - mel range: reference left librosa defaults (sr=22050, fmin=0); paper says 400 Hz-6 kHz.
  - the fit time axis: reference advanced it by the WINDOW length (16 ms) per LMFE frame
    although frames are actually spaced by the 8 ms hop; here it uses the true hop spacing
    (HOP/FS), so slopes are 2x the reference implementation's (but match the paper's math).
"""

import librosa
import numpy as np

FS = 16000
N_FFT = 256        # 16 ms Hamming window
HOP = 128          # 8 ms shift
N_MELS = 40
F_LO = 400
F_HI = 6000
FIT_LEN = 40       # DECAYFITFRLEN: LMFE frames per line fit (320 ms of signal)
FIT_HOP = 2        # fh: fit-window hop in LMFE frames
EPS = 1e-10
MIN_SAMPLES = N_FFT + (FIT_LEN - 1) * HOP  # shortest signal yielding one SDR frame

# slope row of the LLS pseudoinverse; xt spacing = HOP/FS (true frame spacing, see above)
_xt = (np.arange(1, FIT_LEN + 1, dtype=np.float64) * HOP / FS).reshape(-1, 1)
_Q_SLOPE = np.linalg.pinv(np.concatenate([np.ones_like(_xt), _xt], axis=1))[1]

_MEL_BASIS = librosa.filters.mel(sr=FS, n_fft=N_FFT, n_mels=N_MELS, fmin=F_LO, fmax=F_HI)


def lmfe(x):
    """Waveform -> (N_MELS, n_frames) log mel-magnitude energies (natural log)."""
    stft = librosa.stft(x, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                        window="hamming", center=False)
    return np.log(_MEL_BASIS @ np.abs(stft) + EPS)


def compute_decay_rates(x):
    """Waveform (>= MIN_SAMPLES samples, FS) -> (N_MELS, N) float32 SDR sequence."""
    assert len(x) >= MIN_SAMPLES, len(x)
    L = lmfe(x)
    windows = np.lib.stride_tricks.sliding_window_view(L, FIT_LEN, axis=1)[:, ::FIT_HOP, :]
    return (windows @ _Q_SLOPE).astype(np.float32)  # (N_MELS, N)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # frame count matches the reference enframe(): floor((n_lmfe - FIT_LEN + FIT_HOP) / FIT_HOP)
    x = rng.standard_normal(4 * FS)
    n_lmfe = 1 + (len(x) - N_FFT) // HOP
    dr = compute_decay_rates(x)
    assert dr.shape == (N_MELS, (n_lmfe - FIT_LEN + FIT_HOP) // FIT_HOP), dr.shape

    # stationary noise: slopes ~ 0
    assert abs(np.median(dr)) < 1.0, np.median(dr)

    # synthetic free decay with known T60: amplitude envelope 10^(-3 t / T60) (60 dB energy
    # drop at T60). Expected SDR = -3*ln(10)/T60 (log-amplitude).
    # Only the first 0.5 s is measured: later the envelope reaches the EPS log floor.
    for t60 in (0.3, 0.6, 1.0):
        t = np.arange(2 * FS) / FS
        decay = rng.standard_normal(len(t)) * 10 ** (-3 * t / t60)
        n_early = int(0.5 * FS / HOP / FIT_HOP)
        med = np.median(compute_decay_rates(decay)[:, :n_early])
        expect = -3 * np.log(10) / t60
        assert 0.5 < med / expect < 1.5, (t60, med, expect)
        print("T60 %.1f s: median early SDR %+.2f (expected ~%+.2f)" % (t60, med, expect))

    print("features.py self-test OK  (MIN_SAMPLES=%d = %.2f s)" % (MIN_SAMPLES, MIN_SAMPLES / FS))
