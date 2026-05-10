import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import butter, correlate, hilbert, resample_poly, sosfiltfilt

from bk_config import BK_SENSITIVITIES

REC_DIR = Path(__file__).parent / "recordings_p046"
TARGET_FS = 65536
P_REF = 20e-6  # Pa, SPL reference

REF_WAV = REC_DIR / "emo_amusement_freeform.wav"
BK_WAV = REC_DIR / "p005_emo_amusement_freeform_BK_20260428-175252(UTC)-IntegrationRec-0020098323.wav"
RAFA_WAV = REC_DIR / "p005_emo_amusement_freeform_RAFA_20260428-175257(UTC)_Bottom_voiceChat_Omnidirectional.wav"
VOIP_WAV = REC_DIR / "p005_emo_amusement_freeform_VoIP_20260428_195317_to_004915567150671.wav"


def load_wav(path):
    x, fs = sf.read(str(path), always_2d=True)
    return x.astype(np.float64), fs


def resample_to(x, fs_in, fs_out):
    if fs_in == fs_out:
        return x
    g = math.gcd(fs_in, fs_out)
    return resample_poly(x, fs_out // g, fs_in // g)


def bk_to_pascals(x, sens):
    return (x / np.asarray(sens)[None, :]).mean(axis=1)


def envelope(x, fs, lo=300.0, hi=3400.0, env_lp=50.0):
    nyq = fs / 2
    hi = min(hi, 0.95 * nyq)
    sos_bp = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    y = sosfiltfilt(sos_bp, x)
    env = np.abs(hilbert(y))
    sos_lp = butter(4, env_lp / nyq, btype="low", output="sos")
    return sosfiltfilt(sos_lp, env)


def find_lag(ref_env, sig_env):
    a = (ref_env - ref_env.mean()) / (ref_env.std() + 1e-12)
    b = (sig_env - sig_env.mean()) / (sig_env.std() + 1e-12)
    c = correlate(b, a, mode="full")
    return np.argmax(c) - (len(a) - 1)


def align(sig, ref_len, lag):
    out = np.zeros(ref_len)
    if lag >= 0:
        src = sig[lag : lag + ref_len]
        out[: len(src)] = src
    else:
        src = sig[: ref_len + lag]
        out[-lag : -lag + len(src)] = src
    return out


def spl_fast(p_pa, fs, tau=0.125):
    alpha = 1.0 - np.exp(-1.0 / (tau * fs))
    p2 = p_pa.astype(np.float64) ** 2
    msq = np.empty_like(p2)
    acc = 0.0
    for i, v in enumerate(p2):
        acc += alpha * (v - acc)
        msq[i] = acc
    return 10.0 * np.log10(np.maximum(msq, 1e-20) / (P_REF ** 2))


def main():
    sensitivities = BK_SENSITIVITIES
    print("BK sensitivities (V/Pa):", sensitivities)

    ref, fs_ref = load_wav(REF_WAV)
    bk, fs_bk = load_wav(BK_WAV)
    rafa, fs_rafa = load_wav(RAFA_WAV)
    voip, fs_voip = load_wav(VOIP_WAV)

    ref_mono = ref.mean(axis=1)
    bk_pa = bk_to_pascals(bk, sensitivities)
    rafa_mono = rafa.mean(axis=1)
    voip_mono = voip.mean(axis=1)

    ref_rs = resample_to(ref_mono, fs_ref, TARGET_FS)
    bk_rs = resample_to(bk_pa, fs_bk, TARGET_FS)
    rafa_rs = resample_to(rafa_mono, fs_rafa, TARGET_FS)
    voip_rs = resample_to(voip_mono, fs_voip, TARGET_FS)

    ref_env = envelope(ref_rs, TARGET_FS)
    aligned = {}

    for name, sig in [("BK", bk_rs), ("RAFA", rafa_rs), ("VoIP", voip_rs)]:
        env = envelope(sig, TARGET_FS)
        lag = find_lag(ref_env, env)
        aligned[name] = align(sig, len(ref_rs), lag)
        print(f"{name}: lag = {lag} samples ({lag / TARGET_FS:+.4f} s) ")

    t = np.arange(len(ref_rs)) / TARGET_FS

    fig, axes = plt.subplots(5, 1, figsize=(13, 11), sharex=True)

    spl = spl_fast(aligned["BK"], TARGET_FS, tau=0.125)
    peak_spl = float(np.max(spl))
    print(f"BK peak SPL = {peak_spl:6.2f} dB re 20 uPa (Fast, tau=125 ms)")

    axes[0].plot(t, ref_rs, lw=0.5, color="C0")
    axes[0].set_ylabel("Reference\n(norm)")
    axes[0].set_title(f"Time-aligned waveforms @ {TARGET_FS} Hz (envelope cross-correlation)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, aligned["BK"], lw=0.5, color="C0")
    axes[1].set_ylabel("BK (Pa)\n4ch mean")
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, aligned["RAFA"], lw=0.5, color="C0")
    axes[2].set_ylabel("RAFA\n(norm)")
    axes[2].grid(alpha=0.3)

    axes[3].plot(t, aligned["VoIP"], lw=0.5, color="C0")
    axes[3].set_ylabel("VoIP\n(norm)")
    axes[3].grid(alpha=0.3)

    axes[4].plot(t, spl, lw=0.6, color="C0")
    axes[4].set_ylabel("BK SPL\n(dB re 20 µPa)")
    axes[4].set_xlabel("Time (s)")
    axes[4].grid(alpha=0.3)
    axes[4].set_xlim(0.0, t[-1])

    fig.tight_layout()
    plt.show()
    out = Path(__file__).parent / "p046_aligned.png"
    fig.savefig(out, dpi=140)
    print("saved:", out)


if __name__ == "__main__":
    main()
