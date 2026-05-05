import json
from math import gcd
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import butter, correlate, correlation_lags, resample_poly, sosfiltfilt

HERE = Path(__file__).parent
P046 = HERE / "p042_amazment"
REF_PATH = P046 / "emo_amazement_freeform.wav"
BK_STEM = "p196619_p042_emo_amazement_freeform_BK_20260416-151729(UTC)-IntegrationRec-0001671648"
BK_WAV = P046 / f"{BK_STEM}.wav"
BK_JSON = P046 / f"{BK_STEM}.json"
RAFA_PATH = P046 / "p196619_p042_emo_amazement_freeform_RAFA_20260416-151704(UTC)_Bottom_voiceChat_Omnidirectional.wav"
VOIP_PATH = P046 / "p196619_p042_emo_amazement_freeform_VoIP_20260416_171726_to_004915567150671.wav"

P_REF = 20e-6
TRACE_COLOR = "C0"


def load_mono(path: Path):
    data, fs = sf.read(str(path), always_2d=True)
    return data[:, 0].astype(np.float64), fs


def load_bk(wav_path: Path, json_path: Path):
    with json_path.open() as f:
        meta = json.load(f)
    channels = meta["setup"]["channels"]
    sensitivities = np.array([c["transducer"]["sensitivity"] for c in channels])
    serials = [c["transducer"]["serialNumber"] for c in channels]
    data, fs = sf.read(str(wav_path), always_2d=True)
    pressure = data / sensitivities[np.newaxis, :]
    return pressure, fs, serials


def stats(y: np.ndarray, is_bk: bool):
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y * y)))
    spl = 20.0 * np.log10(rms / P_REF) if (is_bk and rms > 0) else None
    return peak, rms, spl


def align_lag(capture: np.ndarray, fs_cap: int, reference: np.ndarray, fs_ref: int,
              bandlimit_hz: float | None = None):
    ref = reference
    if bandlimit_hz is not None:
        sos = butter(8, bandlimit_hz, btype="low", fs=fs_ref, output="sos")
        ref = sosfiltfilt(sos, ref)

    if fs_cap == fs_ref:
        cap_rs = capture
    else:
        g = gcd(int(fs_ref), int(fs_cap))
        cap_rs = resample_poly(capture, up=int(fs_ref) // g, down=int(fs_cap) // g)

    corr = correlate(cap_rs, ref, mode="full", method="fft")
    lags = correlation_lags(len(cap_rs), len(ref), mode="full")
    idx = int(np.argmax(np.abs(corr)))
    lag_seconds = lags[idx] / fs_ref
    denom = float(np.sqrt(np.sum(cap_rs ** 2) * np.sum(ref ** 2)))
    coeff = float(corr[idx] / denom) if denom > 0 else 0.0
    return lag_seconds, coeff


def fmt_title(label: str, fs: int):
    parts = [label, f"fs {fs} Hz"]
    return " — ".join(parts)


def main() -> None:
    ref, fs_ref = load_mono(REF_PATH)
    bk, fs_bk, serials = load_bk(BK_WAV, BK_JSON)
    rafa, fs_rafa = load_mono(RAFA_PATH)
    voip, fs_voip = load_mono(VOIP_PATH)

    rms_per_ch = np.sqrt(np.mean(bk ** 2, axis=0))
    align_ch = int(np.argmax(rms_per_ch))

    lag_bk, coeff_bk = align_lag(bk[:, align_ch], fs_bk, ref, fs_ref)
    lag_rafa, coeff_rafa = align_lag(rafa, fs_rafa, ref, fs_ref)
    lag_voip, coeff_voip = align_lag(voip, fs_voip, ref, fs_ref, bandlimit_hz=3800)

    print("Alignment (lag of capture relative to reference):")
    print(f"  BK   (using Ch{align_ch + 1}): lag = {lag_bk:+.4f} s, corr = {coeff_bk:+.3f}")
    print(f"  RAFA                          : lag = {lag_rafa:+.4f} s, corr = {coeff_rafa:+.3f}")
    print(f"  VoIP (band-limited 3.8 kHz)   : lag = {lag_voip:+.4f} s, corr = {coeff_voip:+.3f}")
    print()

    traces = []

    peak, rms, _ = stats(ref, is_bk=False)
    traces.append({
        "y": ref, "fs": fs_ref, "lag": 0.0,
        "unit": "Amplitude",
        "title": fmt_title("Reference (studio)", fs_ref),
    })

    for c in range(bk.shape[1]):
        y = bk[:, c]
        peak, rms, spl = stats(y, is_bk=True)
        traces.append({
            "y": y, "fs": fs_bk, "lag": lag_bk,
            "unit": "Pressure [Pa]",
            "title": fmt_title(f"BK Ch {c + 1} (SN {serials[c]})", fs_bk),
        })

    peak, rms, _ = stats(rafa, is_bk=False)
    traces.append({
        "y": rafa, "fs": fs_rafa, "lag": lag_rafa,
        "unit": "Amplitude",
        "title": fmt_title("RAFA iPhone bottom mic", fs_rafa),
    })

    peak, rms, _ = stats(voip, is_bk=False)
    traces.append({
        "y": voip, "fs": fs_voip, "lag": lag_voip,
        "unit": "Amplitude",
        "title": fmt_title("VoIP", fs_voip),
    })

    print("Per-trace stats:")
    for tr in traces:
        print(f"  {tr['title']}")
    print()

    for tr in traces:
        tr["t"] = np.arange(tr["y"].size) / tr["fs"] - tr["lag"]
    ref_dur = ref.size / fs_ref

    fig, axes = plt.subplots(len(traces), 1, sharex=True, figsize=(12, 14))
    for ax, tr in zip(axes, traces):
        in_ref_window = (tr["t"] >= 0.0) & (tr["t"] <= ref_dur)
        ax.plot(tr["t"][in_ref_window], tr["y"][in_ref_window],
                linewidth=0.6, color=TRACE_COLOR)
        ax.set_ylabel(tr["unit"])
        ax.set_title(tr["title"], fontsize=9)
        ax.grid(True)
        ax.axvline(0.0, linestyle="--", linewidth=0.5, alpha=0.4, color="k")
        ax.axvline(ref_dur, linestyle="--", linewidth=0.5, alpha=0.4, color="k")
    axes[-1].set_xlabel("Time [s]")
    axes[-1].set_xlim(0.0, ref_dur)
    fig.suptitle("p042 emo_amazement_freeform ")
    fig.tight_layout()
    out_png = HERE / "plot_p046_seven_signals.png"
    fig.savefig(out_png, dpi=120)
    print(f"saved {out_png.name}")

    plt.show()


if __name__ == "__main__":
    main()
