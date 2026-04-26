"""Plot time-domain waveforms of the four emo_adoration_freeform recordings,
with VoIP / RAFA / BK lead-in trimmed so every clip starts at the reference's
first utterance. Includes short-time SPL panels for the three BK channels.
Uses librosa for loading, resampling, and RMS computation."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
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

# B&K Pa = librosa_float * BK_SCALE[ch].
# librosa (via soundfile) normalises 24-bit PCM to float32 in [-1, 1] by dividing
# by 2^23, so float 1.0 == ADC full-scale. BK_SCALE is therefore Pa at full-scale float:
#   BK_SCALE[ch] = V_fullscale / mic_sensitivity
# where V_fullscale = 11.885020 V (from bkdk chunk, Ch1) and sensitivity from JSON.
# Ch1: V (open BNC); Ch2–4: Pa (Type 4951 mics).
BK_SCALE = {
    2: 1969.685454,   # Pa at float full-scale (SN 2475121, sens 0.006033968 V/Pa)
    3: 1808.472201,   # Pa at float full-scale (SN 2475123, sens 0.006571857 V/Pa)
    4: 1753.625713,   # Pa at float full-scale (SN 2475120, sens 0.006777398 V/Pa)
}

COMMON_FS = 16000
VOIP_LP_HZ = 3400


def _prep(x):
    x = x - x.mean()
    rms = np.sqrt(np.mean(x * x))
    if rms > 0:
        x = x / rms
    return x


def _lowpass(x, fs, fc):
    sos = butter(8, fc / (fs / 2), btype="low", output="sos")
    return sosfiltfilt(sos, x)


def find_lag_seconds(ref, fs_ref, dev, fs_dev, lowpass_hz=None):
    """Return the time offset (seconds) into `dev` at which `ref` best aligns."""
    r = librosa.resample(ref.astype(np.float32), orig_sr=fs_ref, target_sr=COMMON_FS).astype(np.float64)
    d = librosa.resample(dev.astype(np.float32), orig_sr=fs_dev, target_sr=COMMON_FS).astype(np.float64)
    if lowpass_hz is not None:
        r = _lowpass(r, COMMON_FS, lowpass_hz)
        d = _lowpass(d, COMMON_FS, lowpass_hz)
    r = _prep(r)
    d = _prep(d)
    corr = correlate(d, r, mode="full", method="fft")
    lag_samples = int(np.argmax(corr)) - (len(r) - 1)
    return lag_samples / COMMON_FS


def trim_to_lag(y, fs, lag_s):
    idx = max(0, int(round(lag_s * fs)))
    return y[idx:], idx / fs


def compute_spl(y_pa, fs, window_s=0.125, hop_s=0.010):
    """Short-time SPL (dB re 20 µPa) via librosa RMS.

    Uses a 125 ms window (IEC 61672 'fast') with a 10 ms hop for a smooth curve.
    Returns (time_seconds, spl_dB) arrays aligned to the start of y_pa.
    """
    frame_length = int(window_s * fs)
    hop_length   = max(1, int(hop_s * fs))
    rms = librosa.feature.rms(
        y=y_pa.astype(np.float32),
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0].astype(np.float64)
    spl   = 20.0 * np.log10(np.maximum(rms, 1e-10) / 20e-6)
    times = librosa.frames_to_time(
        np.arange(len(rms)), sr=fs, hop_length=hop_length
    )
    return times, spl


def main():
    # sr=None keeps native sample rate; mono=True folds to mono for ref/voip/rafa.
    ref,  ref_fs  = librosa.load(str(RECORDINGS["ref"]),  sr=None, mono=True)
    voip, voip_fs = librosa.load(str(RECORDINGS["voip"]), sr=None, mono=True)
    rafa, rafa_fs = librosa.load(str(RECORDINGS["rafa"]), sr=None, mono=True)
    # mono=False returns shape (n_channels, n_samples) — keep all 4 BK channels.
    bk_all, bk_fs = librosa.load(str(RECORDINGS["bk"]), sr=None, mono=False)

    ref  = ref.astype(np.float64)
    voip = voip.astype(np.float64)
    rafa = rafa.astype(np.float64)
    bk_all = bk_all.astype(np.float64)

    # librosa channel index 0-based; BK channel 2 is array row 1, etc.
    bk_ch2 = bk_all[1] * BK_SCALE[2]
    bk_ch3 = bk_all[2] * BK_SCALE[3]
    bk_ch4 = bk_all[3] * BK_SCALE[4]

    lag_voip = find_lag_seconds(ref, ref_fs, voip, voip_fs, lowpass_hz=VOIP_LP_HZ)
    lag_rafa = find_lag_seconds(ref, ref_fs, rafa, rafa_fs)
    lag_bk   = find_lag_seconds(ref, ref_fs, bk_ch2, bk_fs)

    print("Detected lead-in (lag) in each recording:")
    print(f"  VoIP : {lag_voip * 1000:8.1f} ms")
    print(f"  RAFA : {lag_rafa * 1000:8.1f} ms")
    print(f"  BK   : {lag_bk   * 1000:8.1f} ms  (shared by Ch2/Ch3/Ch4)")

    for ch_name, y in (("Ch2", bk_ch2), ("Ch3", bk_ch3), ("Ch4", bk_ch4)):
        peak = np.max(np.abs(y))
        rms  = np.sqrt(np.mean(y ** 2))
        print(f"  BK {ch_name}: peak={peak:6.3f} Pa ({20*np.log10(peak/20e-6):5.1f} dB SPL peak), "
              f"RMS={rms:6.4f} Pa ({20*np.log10(rms/20e-6):5.1f} dB SPL)")

    voip_t, cut_voip = trim_to_lag(voip,   voip_fs, lag_voip)
    rafa_t, cut_rafa = trim_to_lag(rafa,   rafa_fs, lag_rafa)
    bk2_t,  cut_bk   = trim_to_lag(bk_ch2, bk_fs,   lag_bk)
    bk3_t,  _        = trim_to_lag(bk_ch3, bk_fs,   lag_bk)
    bk4_t,  _        = trim_to_lag(bk_ch4, bk_fs,   lag_bk)

    # Compute short-time SPL for each BK channel (125 ms window, 10 ms hop).
    spl2_t, spl2 = compute_spl(bk2_t, bk_fs)
    spl3_t, spl3 = compute_spl(bk3_t, bk_fs)
    spl4_t, spl4 = compute_spl(bk4_t, bk_fs)

    waveform_panels = [
        ("Reference (emo_amazement_freeform)",      ref_fs,  ref,    "Amplitude (norm.)", 0.0),
        ("VoIP call",                              voip_fs, voip_t, "Amplitude (norm.)", cut_voip),
        ("RAFA iPhone Bottom (voiceChat, Omni)",   rafa_fs, rafa_t, "Amplitude (norm.)", cut_rafa),
        ("BK Ch2 (Type 4951, SN 2475121)",         bk_fs,   bk2_t,  "Pressure (Pa)",     cut_bk),
        ("BK Ch3 (Type 4951, SN 2475123)",         bk_fs,   bk3_t,  "Pressure (Pa)",     cut_bk),
        ("BK Ch4 (Type 4951, SN 2475120)",         bk_fs,   bk4_t,  "Pressure (Pa)",     cut_bk),
    ]

    spl_panels = [
        ("BK Ch2 SPL (Type 4951, SN 2475121)", spl2_t, spl2, cut_bk),
        ("BK Ch3 SPL (Type 4951, SN 2475123)", spl3_t, spl3, cut_bk),
        ("BK Ch4 SPL (Type 4951, SN 2475120)", spl4_t, spl4, cut_bk),
    ]

    ref_dur  = ref.shape[0] / ref_fs
    n_rows   = len(waveform_panels) + len(spl_panels)

    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 20), sharex=True)

    # --- Waveform panels ---
    for ax, (label, fs, y, ylabel, cut_s) in zip(axes, waveform_panels):
        t = np.arange(y.shape[0]) / fs
        ax.plot(t, y, lw=0.5)
        title = f"{label}   fs={fs} Hz"
        ax.set_title(title, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.3)

    # --- SPL panels ---
    for ax, (label, t_spl, spl, cut_s) in zip(axes[len(waveform_panels):], spl_panels):
        ax.plot(t_spl, spl, lw=0.8, color="tab:orange")
        title = f"{label}   125 ms window, 10 ms hop"
        ax.set_title(title, fontsize=8)
        ax.set_ylabel("dB SPL\n(re 20 µPa)", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(0, ref_dur)
    fig.suptitle(
        "emo_amazement_freeform",
        y=0.995,
    )
    fig.tight_layout()

    out = HERE / "waveforms_aligned_v3.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
