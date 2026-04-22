"""
RIR extraction from ESS (Exponential Sine Sweep) recording.
Recording: Beyer Dynamics/E300_dp1_mp1.wav
Confirmed structure: 4 sweeps with 2 amplitude levels.
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve, spectrogram
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── PARAMETERS ────────────────────────────────────────────────────────────────
# Confirmed from recording analysis
F1       = 20       # start frequency (Hz) — configured value
F2       = 20_000   # stop  frequency (Hz)
T_SWEEP  = 23   # sweep duration (s) — user-specified; adjust if deconv shows pre-ringing
FS_EXPECTED = 48_000

RECORDING = "Beyer Dynamics/E300_dp1_mp1.wav"

# Confirmed sweep start times in the recording (seconds).
# Each sweep has ~20s active region; we cut T_SWEEP + 3s tail for the reverb.
# Two amplitude groups: loud (sweeps 0,2) and quiet (sweeps 1,3).
SWEEP_STARTS = [
    12.0,   # Sweep 0 — loud  (~-17 dBFS peak)
    34.0,   # Sweep 1 — quiet (~-29 dBFS peak)
    57.0,   # Sweep 2 — loud
    83.0,   # Sweep 3 — quiet
]
TAIL_SEC = 3.0      # seconds of reverb tail to include after each sweep

# ── 1. Load recording ─────────────────────────────────────────────────────────
fs, raw = wavfile.read(RECORDING)
assert fs == FS_EXPECTED, f"Unexpected sample rate {fs}"
print(f"Loaded: {RECORDING}")
print(f"  {fs} Hz | {len(raw)/fs:.2f} s | {raw.dtype}")

dtype_scale = {np.int16: 32768.0, np.int32: 2_147_483_648.0}
y = raw.astype(np.float64) / dtype_scale.get(raw.dtype.type, 1.0)
if y.ndim > 1:
    y = y[:, 0]

# ── 2. Generate reference ESS ─────────────────────────────────────────────────
N_sweep = int(T_SWEEP * fs)
t_ess   = np.arange(N_sweep) / fs
L       = T_SWEEP / np.log(F2 / F1)   # Farina L parameter
ess_ref = np.sin(2 * np.pi * F1 * L * (np.exp(t_ess / L) - 1))

# ── 3. Analytical inverse filter ─────────────────────────────────────────────
# Amplitude envelope compensates for 6 dB/oct spectral tilt of ESS.
# Applied to the original sweep, then time-reversed.
amp_env     = np.exp(-t_ess * np.log(F2 / F1) / T_SWEEP)
inv_filter  = (ess_ref * amp_env)[::-1]
inv_filter /= np.max(np.abs(inv_filter))

# ── 4. Deconvolve each sweep ──────────────────────────────────────────────────
N_extract = int((T_SWEEP + TAIL_SEC) * fs)   # samples to extract per sweep
rirs_raw  = []

for i, t_start in enumerate(SWEEP_STARTS):
    i_start = int(t_start * fs)
    i_end   = min(len(y), i_start + N_extract)
    segment = y[i_start:i_end]

    # Pad if clipped at file end
    if len(segment) < N_extract:
        segment = np.pad(segment, (0, N_extract - len(segment)))

    rir_full = fftconvolve(segment, inv_filter)

    # Peak should be near sample (N_sweep - 1) in the output
    peak_idx = np.argmax(np.abs(rir_full))
    print(f"  Sweep {i}: peak at {peak_idx/fs*1000:.1f} ms  "
          f"(expected ~{(N_sweep-1)/fs*1000:.0f} ms)  "
          f"amplitude={np.abs(rir_full[peak_idx]):.4f}")

    # Trim: keep from a few ms before the peak
    pre = int(0.005 * fs)   # 5 ms guard before direct sound
    s = max(0, peak_idx - pre)
    rirs_raw.append(rir_full[s:s + int(TAIL_SEC * fs)])

# Align all RIRs to same length
min_len = min(len(r) for r in rirs_raw)
rirs = np.stack([r[:min_len] for r in rirs_raw])   # (4, samples)

# ── 5. Average by amplitude group ─────────────────────────────────────────────
# Loud sweeps (0,2) and quiet sweeps (1,3) were at different SPLs.
# Normalise each before averaging so SNR isn't biased.
rirs_norm = rirs / np.max(np.abs(rirs), axis=1, keepdims=True)

rir_loud  = np.mean(rirs_norm[[0, 2]], axis=0)   # sweeps 0 and 2
rir_quiet = np.mean(rirs_norm[[1, 3]], axis=0)   # sweeps 1 and 3
rir_all   = np.mean(rirs_norm, axis=0)            # all four

# Final normalise
rir_all  /= np.max(np.abs(rir_all))
rir_loud /= np.max(np.abs(rir_loud))
rir_quiet/= np.max(np.abs(rir_quiet))

# ── 6. Save ───────────────────────────────────────────────────────────────────
def save_wav(fname, rir, fs):
    out = (rir / np.max(np.abs(rir)) * 0.9 * 32767).astype(np.int16)
    wavfile.write(fname, fs, out)
    print(f"Saved: {fname}  ({len(rir)/fs:.2f} s)")

save_wav("RIR_all_avg.wav",   rir_all,   fs)
save_wav("RIR_loud_avg.wav",  rir_loud,  fs)
save_wav("RIR_quiet_avg.wav", rir_quiet, fs)

# ── 7. Energy Decay Curve helper ─────────────────────────────────────────────
def schroeder_edc(rir):
    sq = rir ** 2
    edc = 10 * np.log10(np.cumsum(sq[::-1])[::-1] / np.sum(sq) + 1e-12)
    return edc

# ── 8. Plot ───────────────────────────────────────────────────────────────────
t_ms = np.arange(min_len) / fs * 1000

fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.35)

# --- (a) Raw recording overview ---
ax0 = fig.add_subplot(gs[0, :])
t_rec = np.arange(len(y)) / fs
ax0.plot(t_rec, y, lw=0.3, color='steelblue')
for i, ts in enumerate(SWEEP_STARTS):
    ax0.axvspan(ts, ts + T_SWEEP, alpha=0.15,
                color='red' if i % 2 == 0 else 'green',
                label=f"Sweep {i} ({'loud' if i%2==0 else 'quiet'})")
ax0.set_xlabel("Time (s)")
ax0.set_ylabel("Amplitude")
ax0.set_title("Full recording — sweep regions highlighted (red=loud, green=quiet)")
handles, labels = ax0.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax0.legend(by_label.values(), by_label.keys(), fontsize=8, loc='upper right')
ax0.grid(True, alpha=0.25)

# --- (b) All four individual RIRs ---
ax1 = fig.add_subplot(gs[1, 0])
for i, rir in enumerate(rirs_norm):
    ax1.plot(t_ms, rir + i * 2.5,
             lw=0.6, label=f"Sweep {i} ({'L' if i%2==0 else 'Q'})")
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("Amplitude (offset)")
ax1.set_title("Individual RIRs (normalised, offset for clarity)")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.25)
ax1.set_xlim([0, t_ms[-1]])

# --- (c) Averaged RIR ---
ax2 = fig.add_subplot(gs[1, 1])
ax2.plot(t_ms, rir_all, lw=0.7, color='black', label='All-avg')
ax2.plot(t_ms, rir_loud, lw=0.7, color='red',   alpha=0.6, label='Loud avg')
ax2.plot(t_ms, rir_quiet, lw=0.7, color='green', alpha=0.6, label='Quiet avg')
ax2.set_xlabel("Time (ms)")
ax2.set_ylabel("Amplitude")
ax2.set_title("Averaged RIR")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.25)
ax2.set_xlim([0, t_ms[-1]])

# --- (d) Energy Decay Curve ---
ax3 = fig.add_subplot(gs[2, 0])
edc_all   = schroeder_edc(rir_all)
edc_loud  = schroeder_edc(rir_loud)
edc_quiet = schroeder_edc(rir_quiet)
ax3.plot(t_ms, edc_all,   lw=1.2, color='black', label='All-avg')
ax3.plot(t_ms, edc_loud,  lw=1.0, color='red',   alpha=0.7, label='Loud avg')
ax3.plot(t_ms, edc_quiet, lw=1.0, color='green', alpha=0.7, label='Quiet avg')
ax3.axhline(-60, color='purple', lw=0.8, ls='--', label='−60 dB (T60)')
ax3.set_ylim([-70, 5])
ax3.set_xlabel("Time (ms)")
ax3.set_ylabel("Level (dB)")
ax3.set_title("Energy Decay Curve (Schroeder)")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.25)
ax3.set_xlim([0, t_ms[-1]])

# --- (e) Spectrogram of averaged RIR ---
ax4 = fig.add_subplot(gs[2, 1])
f_s, t_s, Sxx = spectrogram(rir_all, fs=fs, nperseg=1024, noverlap=896,
                              window='hann')
Sxx_db = 10 * np.log10(Sxx + 1e-12)
pcm = ax4.pcolormesh(t_s * 1000, f_s, Sxx_db,
                     vmin=Sxx_db.max() - 60, vmax=Sxx_db.max(),
                     shading='gouraud', cmap='inferno')
ax4.set_yscale('log')
ax4.set_ylim([50, fs // 2])
ax4.set_xlabel("Time (ms)")
ax4.set_ylabel("Frequency (Hz)")
ax4.set_title("RIR Spectrogram")
plt.colorbar(pcm, ax=ax4, label='dB')
ax4.grid(True, alpha=0.2, color='white')

plt.suptitle(f"Room Impulse Response — E300 dp1 mp1\n"
             f"ESS: f1={F1}Hz f2={F2}Hz T={T_SWEEP}s  fs={fs}Hz",
             fontsize=12, fontweight='bold')

plt.savefig("RIR_analysis.png", dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: RIR_analysis.png")
