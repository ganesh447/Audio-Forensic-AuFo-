import math
from math import gcd
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.colors import Normalize
from scipy.signal import (butter, correlate, correlation_lags, hilbert,
                           resample_poly, sosfiltfilt, spectrogram, welch)

HERE = Path(__file__).parent

# --- Config ---
BK_SENSITIVITIES = [
    0.006033968508228489,
    0.006033968508228489,
    0.006571856617826670,
    0.006777398343872176,
]

REC_DIR  = HERE / "recordings_p046"
TARGET_FS = 65536
P_REF     = 20e-6  # Pa

REF_WAV  = REC_DIR / "emo_amusement_freeform.wav"
BK_WAV   = REC_DIR / "p005_emo_amusement_freeform_BK_20260428-175252(UTC)-IntegrationRec-0020098323.wav"
RAFA_WAV = REC_DIR / "p005_emo_amusement_freeform_RAFA_20260428-175257(UTC)_Bottom_voiceChat_Omnidirectional.wav"
VOIP_WAV = REC_DIR / "p005_emo_amusement_freeform_VoIP_20260428_195317_to_004915567150671.wav"

# --- Spectrogram settings ---
DF_TARGET             = 25.0
WINDOW                = "hann"
F_MIN, F_MAX          =   20.0, 20000.0
SPL_VMIN,  SPL_VMAX   =    0.0,   100.0
DBFS_VMIN, DBFS_VMAX  = -120.0,   -20.0

# ISO 266 nominal 1/3-octave centre frequencies (Hz)
THIRD_OCTAVE_CENTERS_HZ = np.array([
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500,
    630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
    10000, 12500, 16000, 20000,
])


# --- I/O helpers ---
def load_wav(path):
    x, fs = sf.read(str(path), always_2d=True)
    return x.astype(np.float64), fs


def resample_to(x, fs_in, fs_out):
    if fs_in == fs_out:
        return x
    g = math.gcd(fs_in, fs_out)
    return resample_poly(x, fs_out // g, fs_in // g)


def load_mono_resampled(path, fs_target):
    x, fs = load_wav(path)
    mono = x.mean(axis=1)
    return resample_to(mono, fs, fs_target), mono, fs


def load_bk_calibrated(path, sensitivities, fs_target):
    x, fs = load_wav(path)
    pa_ch = x / np.asarray(sensitivities)[None, :]
    pa_mean_rs = resample_to(pa_ch.mean(axis=1), fs, fs_target)
    return pa_ch, pa_mean_rs, fs


# --- Alignment helpers ---
def envelope(x, fs, lo=300.0, hi=3400.0, env_lp=50.0):
    nyq = fs / 2
    hi = min(hi, 0.95 * nyq)
    sos_bp = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos_bp, x)))
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


def align_to_reference(ref_rs, signals, fs):
    ref_env = envelope(ref_rs, fs)
    aligned = {}
    for name, sig in signals.items():
        lag = find_lag(ref_env, envelope(sig, fs))
        aligned[name] = align(sig, len(ref_rs), lag)
    return aligned


def align_lag(capture, fs_cap, ref, fs_ref):
    g = gcd(fs_cap, fs_ref)
    cap_rs  = resample_poly(capture, fs_ref // g, fs_cap // g) if fs_cap != fs_ref else capture
    env_cap = envelope(cap_rs, fs_ref)
    env_ref = envelope(ref, fs_ref)
    env_cap = (env_cap - env_cap.mean()) / (env_cap.std() + 1e-12)
    env_ref = (env_ref - env_ref.mean()) / (env_ref.std() + 1e-12)
    corr = correlate(env_cap, env_ref, mode="full", method="fft")
    lags = correlation_lags(len(env_cap), len(env_ref), mode="full")
    return float(lags[int(np.argmax(np.abs(corr)))]) / fs_ref


def crop_to_ref_window(y, fs, lag_s, ref_dur):
    i0 = max(0, int(round(lag_s * fs)))
    i1 = min(y.size, int(round((lag_s + ref_dur) * fs)))
    return y[i0:i1]


# --- SPL ---
def spl_fast(p_pa, fs, tau=0.125):
    alpha = 1.0 - np.exp(-1.0 / (tau * fs))
    p2  = p_pa.astype(np.float64) ** 2
    msq = np.empty_like(p2)
    acc = 0.0
    for i, v in enumerate(p2):
        acc += alpha * (v - acc)
        msq[i] = acc
    return 10.0 * np.log10(np.maximum(msq, 1e-20) / P_REF ** 2)


# --- Spectrogram ---
def compute_spectrogram(y, fs, unit):
    nperseg = min(max(256, int(round(fs / DF_TARGET))), y.size)
    f, t, Sxx = spectrogram(y, fs=fs, window=WINDOW,
                             nperseg=nperseg, noverlap=nperseg // 2,
                             scaling="density", mode="psd",
                             detrend="constant")
    if unit == "spl":
        db = 10.0 * np.log10(Sxx / P_REF ** 2 + 1e-20)
    else:
        db = 10.0 * np.log10(Sxx + 1e-20)
    return f, t, db


# --- Frequency-domain plots ---
def plot_bk_spectrum(ax, bk_pa_ch, fs, nperseg=8192):
    nperseg = min(nperseg, bk_pa_ch.shape[0])
    for c in range(bk_pa_ch.shape[1]):
        f, Pxx = welch(bk_pa_ch[:, c], fs=fs, window="hann",
                       nperseg=nperseg, noverlap=nperseg // 2,
                       scaling="density", detrend="constant")
        ax.semilogx(f, 10.0 * np.log10(Pxx / P_REF ** 2),
                    lw=0.9, label=f"Ch {c + 1}")
    ax.set_xlim(20.0, fs / 2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB re (20 µPa)²/Hz)")
    ax.set_title("Frequency spectrum - BK channels 1-4")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right", ncols=4)


def third_octave_band_levels(p_pa, fs, centers_hz=THIRD_OCTAVE_CENTERS_HZ, nperseg=8192):
    nperseg = min(nperseg, len(p_pa))
    f, Pxx = welch(p_pa, fs=fs, window="hann",
                   nperseg=nperseg, noverlap=nperseg // 2,
                   scaling="density", detrend="constant")
    df = f[1] - f[0]
    levels = np.full(len(centers_hz), np.nan)
    for i, fc in enumerate(centers_hz):
        f_lo = fc * 2 ** (-1 / 6)
        f_hi = fc * 2 ** (1 / 6)
        if f_hi > fs / 2:
            break
        mask = (f >= f_lo) & (f < f_hi)
        if mask.any():
            band_pa2 = Pxx[mask].sum() * df
            levels[i] = 10.0 * np.log10(band_pa2 / P_REF ** 2)
    return centers_hz, levels


def plot_bk_third_octave(ax, bk_pa_ch, fs, f_min=50.0):
    centers = THIRD_OCTAVE_CENTERS_HZ[THIRD_OCTAVE_CENTERS_HZ >= f_min]
    for c in range(bk_pa_ch.shape[1]):
        fc, lvl = third_octave_band_levels(bk_pa_ch[:, c], fs, centers_hz=centers)
        ax.semilogx(fc, lvl, marker="o", ms=3, lw=0.9, label=f"Ch {c + 1}")
    ax.set_xlim(f_min, min(20000.0, fs / 2))
    ax.set_xticks(centers)
    ax.set_xticklabels(
        [f"{int(f)}" if f >= 100 else f"{f:g}" for f in centers],
        rotation=45, fontsize=7,
    )
    ax.minorticks_off()
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Band SPL\n(dB re 20 µPa)")
    ax.set_title("BK per-channel 1/3-octave band levels")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right", ncols=4)


def plot_normalized_spectrum(ax, sources, nperseg=8192, f_min=20.0):
    f_max = 0.0
    for label, (sig, fs) in sources.items():
        n = min(nperseg, len(sig))
        f, Pxx = welch(sig, fs=fs, window="hann",
                       nperseg=n, noverlap=n // 2,
                       scaling="density", detrend="constant")
        df = f[1] - f[0]
        total = Pxx.sum() * df
        Pxx_norm = Pxx / total
        ax.semilogx(f, 10.0 * np.log10(np.maximum(Pxx_norm, 1e-20)),
                    lw=0.9, label=f"{label} (fs={fs} Hz)")
        f_max = max(f_max, fs / 2)
    ax.set_xlim(f_min, f_max)
    ax.set_ylim(bottom=-80)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized PSD")
    ax.set_title("Frequency spectrum - BK vs RAFA vs VoIP")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right", ncols=3)


def _draw_time_panels(fig, gs, t, ref_rs, aligned, spl, fs_ref, fs_bk, fs_rafa, fs_voip):
    panels = [
        (ref_rs,          f"Reference - fs {fs_ref} Hz",   "Amplitude"),
        (aligned["BK"],   f"BK - fs {fs_bk} Hz",           "Pressure (Pa)"),
        (aligned["RAFA"], f"RAFA iPhone - fs {fs_rafa} Hz", "Amplitude"),
        (aligned["VoIP"], f"VoIP - fs {fs_voip} Hz",        "Amplitude"),
        (spl,             "BK SPL",                         "dB re 20 µPa"),
    ]
    tax = [fig.add_subplot(gs[0])]
    for i in range(1, 5):
        tax.append(fig.add_subplot(gs[i], sharex=tax[0]))
    for ax, (y, title, ylabel) in zip(tax, panels):
        ax.plot(t, y, lw=0.5, color="C0")
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(alpha=0.3)
        plt.setp(ax.get_xticklabels(), visible=False)
    tax[0].set_xlim(0.0, t[-1])
    return tax


def _draw_spectrograms(fig, gs, tax, spec_tracks):
    cmap = plt.get_cmap("magma")
    spec_axes = []
    for row, (label, y, fs, unit, vmin, vmax, cb_label) in enumerate(spec_tracks):
        ax = fig.add_subplot(gs[5 + row], sharex=tax[0])
        spec_axes.append(ax)
        f_sp, t_sp, db = compute_spectrogram(y, fs, unit)
        mesh = ax.pcolormesh(t_sp, f_sp, db, shading="gouraud", cmap=cmap,
                              norm=Normalize(vmin=vmin, vmax=vmax))
        ax.set_yscale("log")
        ax.set_ylim(F_MIN, F_MAX)
        ax.set_facecolor(cmap(0.0))
        ax.set_ylabel("Frequency [Hz]", fontsize=8)
        ax.set_title(f"{label}  —  fs {fs} Hz", fontsize=9)
        ax.grid(True, which="both", alpha=0.2, color="white", linewidth=0.3)
        fig.colorbar(mesh, ax=ax, label=cb_label, pad=0.01, fraction=0.03)
        if row < len(spec_tracks) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
    spec_axes[-1].set_xlabel("Time [s]")
    return spec_axes


def main():
    # --- Load (native rates preserved for spectrograms) ---
    ref_rs,   ref_nat,  fs_ref  = load_mono_resampled(REF_WAV,  TARGET_FS)
    bk_pa_ch, bk_rs,    fs_bk   = load_bk_calibrated(BK_WAV, BK_SENSITIVITIES, TARGET_FS)
    rafa_rs,  rafa_nat, fs_rafa  = load_mono_resampled(RAFA_WAV, TARGET_FS)
    voip_rs,  voip_nat, fs_voip  = load_mono_resampled(VOIP_WAV, TARGET_FS)

    # --- Alignment at TARGET_FS for time panels ---
    aligned = align_to_reference(
        ref_rs, {"BK": bk_rs, "RAFA": rafa_rs, "VoIP": voip_rs}, TARGET_FS
    )
    t       = np.arange(len(ref_rs)) / TARGET_FS
    ref_dur = t[-1]
    spl     = spl_fast(aligned["BK"], TARGET_FS, tau=0.125)

    # --- Alignment at native rates for spectrograms ---
    lag_bk   = align_lag(bk_pa_ch[:, 0], fs_bk,  ref_nat, fs_ref)
    lag_rafa = align_lag(rafa_nat,        fs_rafa, ref_nat, fs_ref)
    lag_voip = align_lag(voip_nat,        fs_voip, ref_nat, fs_ref)
    print(f"BK   lag: {lag_bk:+.4f} s")
    print(f"RAFA lag: {lag_rafa:+.4f} s")
    print(f"VoIP lag: {lag_voip:+.4f} s")

    bk_seg   = crop_to_ref_window(bk_pa_ch[:, 0], fs_bk,  lag_bk,   ref_dur)
    rafa_seg = crop_to_ref_window(rafa_nat,        fs_rafa, lag_rafa, ref_dur)
    voip_seg = crop_to_ref_window(voip_nat,        fs_voip, lag_voip, ref_dur)

    # --- Build figure ---
    fig = plt.figure(figsize=(14, 32), constrained_layout=True)
    gs  = fig.add_gridspec(11, 1, height_ratios=[1, 1, 1, 1, 1, 1.8, 1.8, 1.8, 1.5, 1.5, 1.5])

    tax = _draw_time_panels(fig, gs, t, ref_rs, aligned, spl, fs_ref, fs_bk, fs_rafa, fs_voip)

    spec_tracks = [
        ("BK Ch1",      bk_seg,   fs_bk,   "spl",  SPL_VMIN,  SPL_VMAX,  "dB SPL/Hz"),
        ("RAFA iPhone", rafa_seg, fs_rafa, "dbfs", DBFS_VMIN, DBFS_VMAX, "dBFS/Hz"),
        ("VoIP",        voip_seg, fs_voip, "dbfs", DBFS_VMIN, DBFS_VMAX, "dBFS/Hz"),
    ]
    _draw_spectrograms(fig, gs, tax, spec_tracks)

    plot_bk_spectrum(        fig.add_subplot(gs[8]),  bk_pa_ch, fs_bk)
    plot_bk_third_octave(    fig.add_subplot(gs[9]),  bk_pa_ch, fs_bk)
    plot_normalized_spectrum(fig.add_subplot(gs[10]), {
        "BK Ch1": (bk_pa_ch[:, 0], fs_bk),
        "RAFA":   (rafa_nat,        fs_rafa),
        "VoIP":   (voip_nat,        fs_voip),
    })

    fig.suptitle("p046 emo_amusement_freeform", fontsize=15, fontweight="bold")

    out = HERE / "plot_p046_spectrograms.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out.name}")
    plt.show()


if __name__ == "__main__":
    main()
