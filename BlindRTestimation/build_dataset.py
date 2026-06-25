"""
Faithful implementation of Gamper & Tashev IWAENC 2018 data pipeline.

For each RIR:
  1. Convolve with selected speech clips → reverberant utterance
  2. Chunk into 4s windows with 0.5s hop
  3. Discard silent chunks (RMS > 20dB below full clip RMS)
  4. A-weighted RMS normalisation (paper §3.2)
  5. Gammatone ERB filterbank (21 bands, 400–6000 Hz)
  6. Log frame energy (64-sample frames, 32-sample hop) → 21×N matrix
  7. Spectral whitening (subtract row median)

After all utterances:
  8. Compute global mean and std from all training matrices
  9. Normalise all matrices to ~zero mean, unit std
  10. Save as .npy files + train_manifest.csv
"""

import csv
import random
import shutil 
import sys
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf   
from scipy.signal import fftconvolve, bilinear, lfilter
from tqdm import tqdm
from gammatone.filters import erb_space, make_erb_filters, erb_filterbank


random.seed(42)
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_DIR  = SCRIPT_DIR.parent
RIR_DIR      = PROJECT_DIR / "selected_rirs"
EARS_DIR     = PROJECT_DIR / "Datasets" / "EARS"
OUT_DIR      = SCRIPT_DIR
TRAIN_DIR    = OUT_DIR / "train"
MANIFEST_CSV = OUT_DIR / "train_manifest.csv"
SIM_CSV      = RIR_DIR / "simulated_selected_rirs.csv"
REAL_CSV     = RIR_DIR / "real_selected_rirs.csv"

# ── Config (paper §3.2) ────────────────────────────────────────────────────────

TARGET_SR       = 16_000
WINDOW_SECS     = 4.0
HOP_SECS        = 3.5
WINDOW_SAMPLES  = int(WINDOW_SECS * TARGET_SR)   # 64 000
HOP_SAMPLES     = int(HOP_SECS   * TARGET_SR)    # 56 000
SILENCE_DB      = 20.0      # discard chunk if RMS > 20 dB below full clip
N_BANDS         = 21        # gammatone bands
FREQ_LOW        = 400.0     # Hz
FREQ_HIGH       = 6000.0    # Hz
FRAME_SIZE      = 64        # samples per log-energy frame
FRAME_HOP       = 32        # hop between frames
CLIPS_PER_RIR     = 1    # clips per selected speaker per RIR
SPEAKERS_PER_RIR  = 4    # randomly sample 3 of 10 speakers per RIR → ~32k matrices


# ── A-weighting filter ─────────────────────────────────────────────────────────

def design_a_weighting(fs: int):
    """
    A-weighting IIR filter (IEC 61672), normalised to 0 dB at 1 kHz.
    Returns (b, a) digital filter coefficients for sample rate fs.
    """
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    p = np.pi
    # Analog transfer function: double poles at f1, f4; single poles at f2, f3
    num = [(2*p*f4)**2, 0, 0, 0, 0]
    den = np.polymul([1, 4*p*f4, (2*p*f4)**2], [1, 4*p*f1, (2*p*f1)**2])
    den = np.polymul(np.polymul(den, [1, 2*p*f2]), [1, 2*p*f3])
    b, a = bilinear(num, den, fs)
    # Normalise to 0 dB at 1 kHz
    w_ref = 2 * np.pi * 1000.0 / fs
    z_ref = np.exp(1j * w_ref)
    h_ref = np.polyval(b[::-1], z_ref) / np.polyval(a[::-1], z_ref)
    b = b / abs(h_ref)
    return b, a


# ── Gammatone feature extraction ───────────────────────────────────────────────

def gammatone_log_energy(chunk: np.ndarray) -> np.ndarray:
    filtered = erb_filterbank(chunk.astype(float), _fcoefs).astype(np.float32)

    n_samples = filtered.shape[1]
    n_frames  = (n_samples - FRAME_SIZE) // FRAME_HOP + 1
    features  = np.empty((N_BANDS, n_frames), dtype=np.float32)
    for i in range(n_frames):
        s = i * FRAME_HOP
        frame = filtered[:, s : s + FRAME_SIZE]
        energy = np.sum(frame ** 2, axis=1)
        features[:, i] = np.log(energy + 1e-10)
    return features


def spectral_whiten(feat: np.ndarray) -> np.ndarray:
    """Subtract median from each frequency band (row)."""
    return (feat - np.median(feat, axis=1, keepdims=True)).astype(np.float32)


# ── Audio utilities ────────────────────────────────────────────────────────────

def load_wav(path: Path, target_sr: int) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != target_sr:
        from scipy.signal import resample_poly
        g = gcd(target_sr, sr)
        data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)
    return data


def convolve_rir(speech: np.ndarray, rir: np.ndarray) -> np.ndarray:
    reverberant = fftconvolve(speech, rir)[:len(speech)]
    return reverberant.astype(np.float32)


def rms_db(signal: np.ndarray) -> float:
    rms = np.sqrt(np.mean(signal ** 2))
    return 20 * np.log10(rms + 1e-10)


def chunk_utterance(reverberant: np.ndarray, fs: int, aw_b: np.ndarray, aw_a: np.ndarray) -> list[np.ndarray]:
    """
    Split into 4s / 0.5s-hop chunks; discard silent chunks (paper §3.2).
    Each kept chunk is level-normalised by its A-weighted RMS (paper §3.2).
    """
    clip_rms_db = rms_db(reverberant)
    chunks = []
    start  = 0
    while start + WINDOW_SAMPLES <= len(reverberant):
        chunk = reverberant[start : start + WINDOW_SAMPLES].copy()
        if rms_db(chunk) >= clip_rms_db - SILENCE_DB:
            weighted = lfilter(aw_b, aw_a, chunk)
            rms_w = np.sqrt(np.mean(weighted ** 2))
            if rms_w > 1e-9:
                chunk = chunk / rms_w
                chunks.append(chunk)
        start += HOP_SAMPLES
    return chunks


# ── Load RIR manifests ─────────────────────────────────────────────────────────

def load_manifest(csv_path: Path, rir_subdir: str, t30_col: str) -> list[dict]:
    recs = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            t30 = row.get(t30_col, "").strip()
            if not t30:
                continue
            recs.append({
                "path": RIR_DIR / rir_subdir / row["dest_name"],
                "name": Path(row["dest_name"]).stem,
                "t30":  float(t30),
            })
    return recs


rirs = (
    load_manifest(SIM_CSV,  "simulated", "t30_measured") +
    load_manifest(REAL_CSV, "real",      "T30")
)
random.shuffle(rirs)
print(f"RIRs: {len(rirs)} total (all training)")

# ── Build speech pool ──────────────────────────────────────────────────────────

def is_usable(name: str) -> bool:
    if name.startswith("nonverbal_"):
        return False
    if name.startswith("rainbow_"):
        return True
    if name.startswith("emo_") and name.endswith("_sentences.wav"):
        return True
    return False


speech_pool: dict[str, list[Path]] = {}
for spk_dir in sorted(EARS_DIR.glob("p0*")):
    clips = [f for f in sorted(spk_dir.glob("*.wav")) if is_usable(f.name)]
    if clips:
        speech_pool[spk_dir.name] = clips

speakers = sorted(speech_pool.keys())
print(f"Speakers: {speakers}")

# ── Phase 1: convolve → feature extraction → save raw .npy ────────────────────

aw_b, aw_a = design_a_weighting(TARGET_SR)

_cfs    = erb_space(FREQ_LOW, FREQ_HIGH, N_BANDS)
_fcoefs = make_erb_filters(TARGET_SR, _cfs)

if TRAIN_DIR.exists():
    shutil.rmtree(TRAIN_DIR)
TRAIN_DIR.mkdir(parents=True)

manifest_rows = []
train_sum     = None
train_sq_sum  = None
train_count   = 0
skipped_rir   = 0

for rir_rec in tqdm(rirs, desc="Phase 1: feature extraction"):
    try:
        rir_signal = load_wav(rir_rec["path"], TARGET_SR)
    except Exception as e:
        print(f"  [WARN] RIR load failed {rir_rec['name']}: {e}")
        skipped_rir += 1
        continue

    for speaker in random.sample(speakers, min(SPEAKERS_PER_RIR, len(speakers))):
        clips        = speech_pool[speaker]
        chosen_clips = random.sample(clips, min(CLIPS_PER_RIR, len(clips)))

        for clip_path in chosen_clips:
            try:
                speech = load_wav(clip_path, TARGET_SR)
            except Exception as e:
                print(f"  [WARN] speech load failed {clip_path.name}: {e}")
                continue

            reverberant = convolve_rir(speech, rir_signal)
            chunks      = chunk_utterance(reverberant, TARGET_SR, aw_b, aw_a)

            for w_idx, chunk in enumerate(chunks):
                feat = gammatone_log_energy(chunk)
                feat = spectral_whiten(feat)

                out_name = (
                    f"{rir_rec['name']}__{speaker}_{clip_path.stem}__w{w_idx}.npy"
                )
                out_path = TRAIN_DIR / out_name
                np.save(str(out_path), feat)

                manifest_rows.append({
                    "filename": f"train/{out_name}",
                    "t30":      round(rir_rec["t30"], 4),
                    "speaker":  speaker,
                    "rir":      rir_rec["name"],
                    "shape":    f"{feat.shape[0]}x{feat.shape[1]}",
                })

                if train_sum is None:
                    train_sum    = np.zeros_like(feat, dtype=np.float64)
                    train_sq_sum = np.zeros_like(feat, dtype=np.float64)
                train_sum    += feat
                train_sq_sum += feat ** 2
                train_count  += 1

print(f"\nPhase 1 done — {len(manifest_rows)} feature matrices")
print(f"Skipped RIRs: {skipped_rir}")

# ── Phase 2: global normalisation using training stats ─────────────────────────

global_mean = (train_sum / train_count).astype(np.float32)
global_std  = np.sqrt(
    np.maximum(train_sq_sum / train_count - (train_sum / train_count) ** 2, 1e-10)
).astype(np.float32)

print(f"\nGlobal mean range : {global_mean.min():.4f} – {global_mean.max():.4f}")
print(f"Global std  range : {global_std.min():.4f} – {global_std.max():.4f}")

np.save(str(OUT_DIR / "global_mean.npy"), global_mean)
np.save(str(OUT_DIR / "global_std.npy"),  global_std)
print("Saved global_mean.npy and global_std.npy")

for row in tqdm(manifest_rows, desc="Phase 2: normalising"):
    path = OUT_DIR / row["filename"]
    feat = np.load(str(path))
    feat = (feat - global_mean) / (global_std + 1e-8)
    np.save(str(path), feat.astype(np.float32))

# ── Save manifest ──────────────────────────────────────────────────────────────

fieldnames = ["filename", "t30", "speaker", "rir", "shape"]
with open(MANIFEST_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(manifest_rows)

print(f"\nDone.")
print(f"  Train matrices : {len(manifest_rows)}")
print(f"  Feature shape  : {manifest_rows[0]['shape']} per matrix")
print(f"  Manifest       : {MANIFEST_CSV}")
