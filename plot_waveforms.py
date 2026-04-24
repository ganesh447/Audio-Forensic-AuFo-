"""Plot time-domain waveforms of the four emo_adoration_freeform recordings."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

HERE = Path(__file__).resolve().parent
REC = HERE / "recordings"

RECORDINGS = {
    "Reference (emo_adoration_freeform)": REC / "emo_adoration_freeform.wav",
    "VoIP call": REC / "p196617_p042_emo_adoration_freeform_VoIP_20260416_171506_to_004915567150671.wav",
    "RAFA iPhone Bottom (voiceChat, Omni)": REC / "p195198_p042_emo_adoration_freeform_RAFA_20260416-151232(UTC)_Bottom_voiceChat_Omnidirectional.wav",
    "BK LAN-XI 4-channel": REC / "p196617_p042_emo_adoration_freeform_BK_20260416-151508(UTC)-IntegrationRec-0001530997.wav",
}

# BK calibration from the JSON sidecar
BK_RANGE_VPEAK = 10.0
BK_SENS = {
    2: 0.006033968508228489,  # V/Pa, SN 2475121
    3: 0.006571856617826670,  # V/Pa, SN 2475123
    4: 0.006777398343872176,  # V/Pa, SN 2475120
}


def load_wav(path):
    fs, data = wavfile.read(str(path))
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float64) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float64)
    return fs, data


def main():
    ref_fs, ref = load_wav(RECORDINGS["Reference (emo_adoration_freeform)"])
    voip_fs, voip = load_wav(RECORDINGS["VoIP call"])
    rafa_fs, rafa = load_wav(RECORDINGS["RAFA iPhone Bottom (voiceChat, Omni)"])
    bk_fs, bk = load_wav(RECORDINGS["BK LAN-XI 4-channel"])

    if ref.ndim > 1:
        ref = ref[:, 0]
    if voip.ndim > 1:
        voip = voip[:, 0]
    if rafa.ndim > 1:
        rafa = rafa[:, 0]

    # BK channels are 1..4 in JSON; numpy columns are 0..3
    bk_v = bk * BK_RANGE_VPEAK
    bk_ch2_pa = bk_v[:, 1] / BK_SENS[2]
    bk_ch3_pa = bk_v[:, 2] / BK_SENS[3]
    bk_ch4_pa = bk_v[:, 3] / BK_SENS[4]

    panels = [
        ("Reference (emo_adoration_freeform)", ref_fs, ref, "Amplitude (norm.)"),
        ("VoIP call", voip_fs, voip, "Amplitude (norm.)"),
        ("RAFA iPhone Bottom (voiceChat, Omni)", rafa_fs, rafa, "Amplitude (norm.)"),
        ("BK Ch2 (Type 4951, SN 2475121)", bk_fs, bk_ch2_pa, "Pressure (Pa)"),
        ("BK Ch3 (Type 4951, SN 2475123)", bk_fs, bk_ch3_pa, "Pressure (Pa)"),
        ("BK Ch4 (Type 4951, SN 2475120)", bk_fs, bk_ch4_pa, "Pressure (Pa)"),
    ]

    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 14), sharex=True)
    for ax, (label, fs, y, ylabel) in zip(axes, panels):
        t = np.arange(y.shape[0]) / fs
        ax.plot(t, y, lw=0.5)
        ax.set_title(f"{label}   fs={fs} Hz, dur={y.shape[0] / fs:.2f} s")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("emo_adoration_freeform — time-domain waveforms", y=0.995)
    fig.tight_layout()

    out = HERE / "waveforms.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
