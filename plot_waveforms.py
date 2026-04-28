"""Plot time-domain waveforms of the four emo_amazement_freeform recordings,
with VoIP / RAFA / BK lead-in trimmed so every clip starts at the reference's
first utterance. Includes short-time SPL panels for the three BK channels.
All signals are resampled to TARGET_FS at load time."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import librosa
from scipy.signal import correlate, butter, sosfiltfilt

HERE = Path(__file__).resolve().parent
REC = HERE / "recordings"

RECORDINGS = {
    "ref":  REC / "emo_amazement_freeform.wav",
    "voip": REC / "p196619_p042_emo_amazement_freeform_VoIP_20260416_171726_to_004915567150671.wav",
    "rafa": REC / "p196619_p042_emo_amazement_freeform_RAFA_20260416-151704(UTC)_Bottom_voiceChat_Omnidirectional.wav",
    "bk":   REC / "p196619_p042_emo_amazement_freeform_BK_20260416-151729(UTC)-IntegrationRec-0001671648.wav",
}

TARGET_FS  = 65536   # all signals resampled here at load time; edit to change fs
VOIP_LP_HZ = 3400

# B&K engineering-unit = librosa_float * BK_SCALE[ch].
# The bkdk chunk stores scale factors at *int32* full-scale (2^31).
# librosa (via soundfile) normalises 24-bit PCM to float by dividing by 2^23.
# Correction factor: bkdk_raw / 256  (= 2^23 / 2^31 = 1/256 applied to numerator).
#   bkdk raw values: Ch1=11.885020, Ch2=1969.685454, Ch3=1808.472201, Ch4=1753.625713
BK_SCALE = {
    1: 11.885020   / 256,   # ≈ 0.04642 V  at float full-scale (open BNC, no transducer)
    2: 1969.685454 / 256,   # ≈ 7.6941  Pa at float full-scale (SN 2475121, 0.006034 V/Pa)
    3: 1808.472201 / 256,   # ≈ 7.0643  Pa at float full-scale (SN 2475123, 0.006572 V/Pa)
    4: 1753.625713 / 256,   # ≈ 6.8501  Pa at float full-scale (SN 2475120, 0.006777 V/Pa)
}


def _resample_to_target(y, fs_in):
    """Resample y (1-D or 2-D channels×samples) to TARGET_FS if needed."""
    if fs_in == TARGET_FS:
        return y.astype(np.float64)
    return librosa.resample(
        y.astype(np.float32), orig_sr=fs_in, target_sr=TARGET_FS
    ).astype(np.float64)


def _prep(x):
    x = x - x.mean()
    rms = np.sqrt(np.mean(x * x))
    if rms > 0:
        x = x / rms
    return x


def _lowpass(x, fc):
    sos = butter(8, fc / (TARGET_FS / 2), btype="low", output="sos")
    return sosfiltfilt(sos, x)


def find_lag_seconds(ref, dev, lowpass_hz=None):
    """Return the time offset (seconds) into `dev` at which `ref` best aligns.
    Both signals must already be at TARGET_FS."""
    r = ref.copy()
    d = dev.copy()
    if lowpass_hz is not None:
        r = _lowpass(r, lowpass_hz)
        d = _lowpass(d, lowpass_hz)
    r = _prep(r)
    d = _prep(d)
    corr = correlate(d, r, mode="full", method="fft")
    lag_samples = int(np.argmax(corr)) - (len(r) - 1)
    return lag_samples / TARGET_FS


def trim_to_lag(y, lag_s):
    idx = max(0, int(round(lag_s * TARGET_FS)))
    return y[idx:], idx / TARGET_FS


def compute_spl(y_pa, window_s=0.125, hop_s=0.010):
    """Short-time SPL (dB re 20 µPa) via librosa RMS.

    125 ms window (IEC 61672 'fast'), 10 ms hop.
    Returns (time_seconds, spl_dB) aligned to the start of y_pa.
    """
    frame_length = int(window_s * TARGET_FS)
    hop_length   = max(1, int(hop_s * TARGET_FS))
    rms = librosa.feature.rms(
        y=y_pa.astype(np.float32),
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0].astype(np.float64)
    spl   = 20.0 * np.log10(np.maximum(rms, 1e-10) / 20e-6)
    times = librosa.frames_to_time(
        np.arange(len(rms)), sr=TARGET_FS, hop_length=hop_length
    )
    return times, spl



def main():
    # Load at native sample rates, then resample everything to TARGET_FS.
    ref,    ref_fs  = librosa.load(str(RECORDINGS["ref"]),  sr=None, mono=True)
    voip,   voip_fs = librosa.load(str(RECORDINGS["voip"]), sr=None, mono=True)
    rafa,   rafa_fs = librosa.load(str(RECORDINGS["rafa"]), sr=None, mono=True)
    bk_all, bk_fs   = librosa.load(str(RECORDINGS["bk"]),  sr=None, mono=False)

    ref    = _resample_to_target(ref,    ref_fs)
    voip   = _resample_to_target(voip,   voip_fs)
    rafa   = _resample_to_target(rafa,   rafa_fs)
    bk_all = _resample_to_target(bk_all, bk_fs)

    # librosa channel index 0-based; BK channel N is array row N-1.
    bk_ch1 = bk_all[0] * BK_SCALE[1]
    bk_ch2 = bk_all[1] * BK_SCALE[2]
    bk_ch3 = bk_all[2] * BK_SCALE[3]
    bk_ch4 = bk_all[3] * BK_SCALE[4]

    lag_voip = find_lag_seconds(ref, voip,   lowpass_hz=VOIP_LP_HZ)
    lag_rafa = find_lag_seconds(ref, rafa)
    lag_bk   = find_lag_seconds(ref, bk_ch2)

    print(f"All signals resampled to {TARGET_FS} Hz")
    print("Detected lead-in (lag) in each recording:")
    print(f"  VoIP : {lag_voip * 1000:8.1f} ms")
    print(f"  RAFA : {lag_rafa * 1000:8.1f} ms")
    print(f"  BK   : {lag_bk   * 1000:8.1f} ms  (shared by Ch1/Ch2/Ch3/Ch4)")

    for ch_name, y in (("Ch2", bk_ch2), ("Ch3", bk_ch3), ("Ch4", bk_ch4)):
        peak = np.max(np.abs(y))
        rms  = np.sqrt(np.mean(y ** 2))
        print(f"  BK {ch_name}: peak={peak:.4e} Pa ({20*np.log10(peak/20e-6):5.1f} dB SPL peak), "
              f"RMS={rms:.4e} Pa ({20*np.log10(rms/20e-6):5.1f} dB SPL)")

    ref_n = ref.shape[0]

    voip_t, cut_voip = trim_to_lag(voip,   lag_voip);  voip_t = voip_t[:ref_n]
    rafa_t, cut_rafa = trim_to_lag(rafa,   lag_rafa);  rafa_t = rafa_t[:ref_n]
    bk1_t,  _        = trim_to_lag(bk_ch1, lag_bk);   bk1_t  = bk1_t[:ref_n]
    bk2_t,  cut_bk   = trim_to_lag(bk_ch2, lag_bk);   bk2_t  = bk2_t[:ref_n]
    bk3_t,  _        = trim_to_lag(bk_ch3, lag_bk);   bk3_t  = bk3_t[:ref_n]
    bk4_t,  _        = trim_to_lag(bk_ch4, lag_bk);   bk4_t  = bk4_t[:ref_n]

    spl2_t, spl2 = compute_spl(bk2_t)
    spl3_t, spl3 = compute_spl(bk3_t)
    spl4_t, spl4 = compute_spl(bk4_t)

    waveform_panels = [
        ("Reference (emo_amazement_freeform)",      ref,    "Amplitude (norm.)", 0.0),
        ("VoIP call",                               voip_t, "Amplitude (norm.)", cut_voip),
        ("RAFA iPhone Bottom (voiceChat, Omni)",    rafa_t, "Amplitude (norm.)", cut_rafa),
        ("BK Ch1 (open BNC, voltage)",              bk1_t,  "Voltage (V)",       cut_bk),
        ("BK Ch2 (Type 4951, SN 2475121)",          bk2_t,  "Pressure (Pa)",     cut_bk),
        ("BK Ch3 (Type 4951, SN 2475123)",          bk3_t,  "Pressure (Pa)",     cut_bk),
        ("BK Ch4 (Type 4951, SN 2475120)",          bk4_t,  "Pressure (Pa)",     cut_bk),
    ]

    spl_panels = [
        ("BK Ch2 SPL (Type 4951, SN 2475121)", spl2_t, spl2),
        ("BK Ch3 SPL (Type 4951, SN 2475123)", spl3_t, spl3),
        ("BK Ch4 SPL (Type 4951, SN 2475120)", spl4_t, spl4),
    ]

    ref_dur = ref.shape[0] / TARGET_FS
    n_rows  = len(waveform_panels) + len(spl_panels)

    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 22), sharex=True)

    pa_peak = max(np.max(np.abs(bk2_t)), np.max(np.abs(bk3_t)), np.max(np.abs(bk4_t)))
    pa_lim  = pa_peak * 1.1

    for ax, (label, y, ylabel, _cut) in zip(axes, waveform_panels):
        t = np.arange(len(y)) / TARGET_FS
        ax.plot(t, y, lw=0.5)
        ax.set_title(f"{label}   fs={TARGET_FS} Hz", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.3)
        if ylabel == "Pressure (Pa)":
            ax.set_ylim(-pa_lim, pa_lim)

    for ax, (label, t_spl, spl) in zip(axes[len(waveform_panels):], spl_panels):
        ax.plot(t_spl, spl, lw=0.8, color="tab:orange")
        ax.set_title(f"{label}   125 ms window, 10 ms hop", fontsize=8)
        ax.set_ylabel("dB SPL\n(re 20 µPa)", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_locator(MultipleLocator(5))

    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(0, ref_dur)
    fig.suptitle("emo_amazement_freeform", y=0.995)
    fig.tight_layout()

    out = HERE / "waveforms_aligned_v3.png" 
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
