from math import gcd
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.signal import (butter, correlate, correlation_lags, hilbert,
                           resample_poly, sosfiltfilt, spectrogram)

from bk_config import BK_SENSITIVITIES
from plot_p046 import (
    TARGET_FS, P_REF,
    REF_WAV, BK_WAV, RAFA_WAV, VOIP_WAV,
    load_mono_resampled, load_bk_calibrated,
    align_to_reference, spl_fast,
)

HERE = Path(__file__).parent

DF_TARGET             = 25.0
WINDOW                = "hann"
F_MIN, F_MAX          =   20.0, 20000.0
SPL_VMIN,  SPL_VMAX   =    0.0,   100.0
DBFS_VMIN, DBFS_VMAX  = -120.0,   -20.0


def _resample(x, fs_in, fs_out):
    if fs_in == fs_out:
        return x
    g = gcd(fs_in, fs_out)
    return resample_poly(x, fs_out // g, fs_in // g)


def _envelope(x, fs, lo=300.0, hi=3400.0, env_lp=50.0):
    nyq = fs / 2.0
    hi = min(hi, 0.95 * nyq)
    sos_bp = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos_bp, x)))
    sos_lp = butter(4, env_lp / nyq, btype="low", output="sos")
    return sosfiltfilt(sos_lp, env)


def align_lag(capture, fs_cap, ref, fs_ref):
    cap_rs  = _resample(capture, fs_cap, fs_ref)
    env_cap = _envelope(cap_rs, fs_ref)
    env_ref = _envelope(ref, fs_ref)
    env_cap = (env_cap - env_cap.mean()) / (env_cap.std() + 1e-12)
    env_ref = (env_ref - env_ref.mean()) / (env_ref.std() + 1e-12)
    corr = correlate(env_cap, env_ref, mode="full", method="fft")
    lags = correlation_lags(len(env_cap), len(env_ref), mode="full")
    return float(lags[int(np.argmax(np.abs(corr)))]) / fs_ref


def crop_to_ref_window(y, fs, lag_s, ref_dur):
    i0 = max(0, int(round(lag_s * fs)))
    i1 = min(y.size, int(round((lag_s + ref_dur) * fs)))
    return y[i0:i1]


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


def main():
    # Load everything once — native rates preserved for spectrograms
    ref_rs,   ref_nat,  fs_ref  = load_mono_resampled(REF_WAV,  TARGET_FS)
    bk_pa_ch, bk_rs,    fs_bk   = load_bk_calibrated(BK_WAV, BK_SENSITIVITIES, TARGET_FS)
    rafa_rs,  rafa_nat, fs_rafa  = load_mono_resampled(RAFA_WAV, TARGET_FS)
    voip_rs,  voip_nat, fs_voip  = load_mono_resampled(VOIP_WAV, TARGET_FS)

    # Time-domain alignment at TARGET_FS (shared grid for time panels)
    aligned = align_to_reference(
        ref_rs, {"BK": bk_rs, "RAFA": rafa_rs, "VoIP": voip_rs}, TARGET_FS
    )
    t       = np.arange(len(ref_rs)) / TARGET_FS
    ref_dur = t[-1]
    spl     = spl_fast(aligned["BK"], TARGET_FS, tau=0.125)

    # Spectrogram alignment at native rates (preserves full bandwidth per device)
    lag_bk   = align_lag(bk_pa_ch[:, 0], fs_bk,  ref_nat, fs_ref)
    lag_rafa = align_lag(rafa_nat,        fs_rafa, ref_nat, fs_ref)
    lag_voip = align_lag(voip_nat,        fs_voip, ref_nat, fs_ref)
    print(f"BK   lag: {lag_bk:+.4f} s")
    print(f"RAFA lag: {lag_rafa:+.4f} s")
    print(f"VoIP lag: {lag_voip:+.4f} s")

    bk_seg   = crop_to_ref_window(bk_pa_ch[:, 0], fs_bk,  lag_bk,   ref_dur)
    rafa_seg = crop_to_ref_window(rafa_nat,        fs_rafa, lag_rafa, ref_dur)
    voip_seg = crop_to_ref_window(voip_nat,        fs_voip, lag_voip, ref_dur)

    # --- Figure: 5 time panels + 3 spectrograms, all sharing x-axis ---
    fig = plt.figure(figsize=(14, 22), constrained_layout=True)
    gs  = fig.add_gridspec(8, 1, height_ratios=[1, 1, 1, 1, 1, 1.8, 1.8, 1.8])

    time_panels = [
        (ref_rs,          f"Reference - fs  {fs_ref} Hz",    "Amplitude"),
        (aligned["BK"],   f"BK - fs  {fs_bk} Hz",            "Pressure (Pa)"),
        (aligned["RAFA"], f"RAFA iPhone - fs  {fs_rafa} Hz",  "Amplitude"),
        (aligned["VoIP"], f"VoIP - fs  {fs_voip} Hz",         "Amplitude"),
        (spl,             "BK SPL ",        "dB re 20 µPa"),
    ]

    tax = [fig.add_subplot(gs[0])]
    for i in range(1, 5):
        tax.append(fig.add_subplot(gs[i], sharex=tax[0]))
    for ax, (y, title, ylabel) in zip(tax, time_panels):
        ax.plot(t, y, lw=0.5, color="C0")
        ax.set_title(title, fontsize=9, loc="center")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(alpha=0.3)
    for ax in tax:
        plt.setp(ax.get_xticklabels(), visible=False)

    spec_tracks = [
        ("BK Ch1",      bk_seg,   fs_bk,   "spl",  SPL_VMIN,  SPL_VMAX,  "dB SPL/Hz"),
        ("RAFA iPhone", rafa_seg, fs_rafa, "dbfs", DBFS_VMIN, DBFS_VMAX, "dBFS/Hz"),
        ("VoIP",        voip_seg, fs_voip, "dbfs", DBFS_VMIN, DBFS_VMAX, "dBFS/Hz"),
    ]
    cmap = plt.get_cmap("magma")
    for row, (label, y, fs, unit, vmin, vmax, cb_label) in enumerate(spec_tracks):
        ax = fig.add_subplot(gs[5 + row], sharex=tax[0])
        f_sp, t_sp, db = compute_spectrogram(y, fs, unit)
        mesh = ax.pcolormesh(t_sp, f_sp, db, shading="gouraud", cmap=cmap,
                              norm=Normalize(vmin=vmin, vmax=vmax))
        ax.set_yscale("log")
        ax.set_ylim(F_MIN, F_MAX)
        ax.set_facecolor(cmap(0.0))
        ax.set_ylabel("Frequency [Hz]", fontsize=8)
        ax.grid(True, which="both", alpha=0.2, color="white", linewidth=0.3)
        nyq   = fs / 2.0
        title = f"{label}  —  fs {fs} Hz"
        ax.set_title(title, fontsize=9)
        fig.colorbar(mesh, ax=ax, label=cb_label, pad=0.01, fraction=0.03)
        if row < 2:
            plt.setp(ax.get_xticklabels(), visible=False)

    tax[0].set_xlim(0.0, ref_dur)
    fig.axes[-1].set_xlabel("Time [s]")
    fig.suptitle(
        "p046 emo_amusement_freeform — Time domain & Spectrograms  (BK · RAFA · VoIP)",
        fontsize=11, fontweight="bold",
    )

    out = HERE / "plot_p046_spectrograms.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out.name}")
    plt.show()


if __name__ == "__main__":
    main()
