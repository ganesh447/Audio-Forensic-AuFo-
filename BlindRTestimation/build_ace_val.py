"""
Build val/*.npy feature matrices from the ACE dev dataset.

Uses 12 ACE single-channel RIRs (Lecture_Room_2 excluded — T30 > 1.5s)
convolved with ACE anechoic speech clips (s3, s4, s5 only — s1/s2 too short).

MUST run after build_dataset.py — loads global_mean.npy and global_std.npy
produced by training, does NOT recompute them.

Pipeline per chunk is identical to build_dataset.py:
  convolve → chunk (4s/0.5s) → silence rejection → A-weighted RMS norm
  → gammatone filterbank → log frame energy → spectral whiten → z-score norm
"""

import csv
import os
import random
import sys
from math import gcd

random.seed(42)

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, bilinear, lfilter, resample_poly
from tqdm import tqdm
from gammatone.filters import erb_space, make_erb_filters, erb_filterbank

# ── Paths ──────────────────────────────────────────────────────────────────────

ACE_RIR_ROOT  = r"C:\Users\Sai\Desktop\main_aufo\plot_task5\Datasets\ACE\Single"
ACE_SPEECH    = r"C:\Users\Sai\Desktop\main_aufo\plot_task5\Datasets\ACE\Speech"
CNN_DIR       = r"C:\Users\Sai\Desktop\main_aufo\plot_task5\CNN_T60"
VAL_DIR       = os.path.join(CNN_DIR, "val")
VAL_MANIFEST  = os.path.join(CNN_DIR, "val_manifest.csv")
MEAN_PATH     = os.path.join(CNN_DIR, "global_mean.npy")
STD_PATH      = os.path.join(CNN_DIR, "global_std.npy")

# ── ACE ground truth T30 — 12 RIRs, Lecture_Room_2 excluded (T30 > 1.5s) ──────

ACE_RIRS = [
    {"file": "Single/Building_Lobby/1/Single_EE_lobby_1_RIR.wav", "room": "Building_Lobby",  "pos": 1, "t30": 0.846},
    {"file": "Single/Building_Lobby/2/Single_EE_lobby_2_RIR.wav", "room": "Building_Lobby",  "pos": 2, "t30": 0.861},
    {"file": "Single/Lecture_Room_1/1/Single_508_1_RIR.wav",       "room": "Lecture_Room_1",  "pos": 1, "t30": 0.732},
    {"file": "Single/Lecture_Room_1/2/Single_508_2_RIR.wav",       "room": "Lecture_Room_1",  "pos": 2, "t30": 0.729},
    {"file": "Single/Meeting_Room_1/1/Single_503_1_RIR.wav",       "room": "Meeting_Room_1",  "pos": 1, "t30": 0.482},
    {"file": "Single/Meeting_Room_1/2/Single_503_2_RIR.wav",       "room": "Meeting_Room_1",  "pos": 2, "t30": 0.483},
    {"file": "Single/Meeting_Room_2/1/Single_611_1_RIR.wav",       "room": "Meeting_Room_2",  "pos": 1, "t30": 0.422},
    {"file": "Single/Meeting_Room_2/2/Single_611_2_RIR.wav",       "room": "Meeting_Room_2",  "pos": 2, "t30": 0.398},
    {"file": "Single/Office_1/1/Single_502_1_RIR.wav",             "room": "Office_1",        "pos": 1, "t30": 0.462},
    {"file": "Single/Office_1/2/Single_502_2_RIR.wav",             "room": "Office_1",        "pos": 2, "t30": 0.350},
    {"file": "Single/Office_2/1/Single_803_1_RIR.wav",             "room": "Office_2",        "pos": 1, "t30": 0.413},
    {"file": "Single/Office_2/2/Single_803_2_RIR.wav",             "room": "Office_2",        "pos": 2, "t30": 0.415},
]

# Resolve to absolute paths
ACE_ROOT = r"C:\Users\Sai\Desktop\main_aufo\plot_task5\Datasets\ACE"
for rec in ACE_RIRS:
    rec["path"] = os.path.join(ACE_ROOT, rec["file"])

# ── Config — must match build_dataset.py exactly ──────────────────────────────

TARGET_SR      = 16_000
WINDOW_SECS    = 4.0
HOP_SECS       = 0.5
WINDOW_SAMPLES = int(WINDOW_SECS * TARGET_SR)   # 64,000
HOP_SAMPLES    = int(HOP_SECS   * TARGET_SR)    # 8,000
SILENCE_DB     = 20.0
N_BANDS        = 21
FREQ_LOW       = 400.0
FREQ_HIGH      = 6000.0
FRAME_SIZE     = 64
FRAME_HOP      = 32
MIN_DURATION   = 4.0   # seconds — skip clips shorter than one window

# ── A-weighting filter (identical to build_dataset.py) ────────────────────────

def design_a_weighting(fs):
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    p = np.pi
    num = [(2*p*f4)**2, 0, 0, 0, 0]
    den = np.polymul([1, 4*p*f4, (2*p*f4)**2], [1, 4*p*f1, (2*p*f1)**2])
    den = np.polymul(np.polymul(den, [1, 2*p*f2]), [1, 2*p*f3])
    b, a = bilinear(num, den, fs)
    w_ref = 2 * np.pi * 1000.0 / fs
    z_ref = np.exp(1j * w_ref)
    h_ref = np.polyval(b[::-1], z_ref) / np.polyval(a[::-1], z_ref)
    return b / abs(h_ref), a

# ── Gammatone feature extraction (identical to build_dataset.py) ──────────────

def gammatone_log_energy(chunk):
    filtered = erb_filterbank(chunk.astype(float), _fcoefs).astype(np.float32)
    n_frames = (filtered.shape[1] - FRAME_SIZE) // FRAME_HOP + 1
    features = np.empty((N_BANDS, n_frames), dtype=np.float32)
    for i in range(n_frames):
        s = i * FRAME_HOP
        energy = np.sum(filtered[:, s:s + FRAME_SIZE] ** 2, axis=1)
        features[:, i] = np.log(energy + 1e-10)
    return features

def spectral_whiten(feat):
    return (feat - np.median(feat, axis=1, keepdims=True)).astype(np.float32)

# ── Audio utilities ────────────────────────────────────────────────────────────

def load_wav(path):
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        g    = gcd(sr, TARGET_SR)
        data = resample_poly(data, TARGET_SR // g, sr // g).astype(np.float32)
    return data

def rms_db(signal):
    return 20 * np.log10(np.sqrt(np.mean(signal ** 2)) + 1e-10)

def convolve_rir(speech, rir):
    return fftconvolve(speech, rir)[:len(speech)].astype(np.float32)

def chunk_utterance(reverberant, aw_b, aw_a):
    clip_rms = rms_db(reverberant)
    chunks, start = [], 0
    while start + WINDOW_SAMPLES <= len(reverberant):
        chunk = reverberant[start:start + WINDOW_SAMPLES].copy()
        if rms_db(chunk) >= clip_rms - SILENCE_DB:
            weighted = lfilter(aw_b, aw_a, chunk)
            rms_w    = np.sqrt(np.mean(weighted ** 2))
            if rms_w > 1e-9:
                chunk = chunk / rms_w
            chunks.append(chunk)
        start += HOP_SAMPLES
    return chunks

# ── Load training normalisation stats ─────────────────────────────────────────

if not os.path.exists(MEAN_PATH) or not os.path.exists(STD_PATH):
    sys.exit("ERROR: global_mean.npy / global_std.npy not found. "
             "Run build_dataset.py first.")

global_mean = np.load(MEAN_PATH)
global_std  = np.load(STD_PATH)
print(f"Loaded global_mean {global_mean.shape} and global_std from {CNN_DIR}")

# ── Collect usable ACE speech clips (s3, s4, s5 only) ─────────────────────────

speech_clips = []
for fname in sorted(os.listdir(ACE_SPEECH)):
    if not fname.endswith(".wav"):
        continue
    # filename pattern: F1s3.wav, M4s4.wav etc.
    stem   = os.path.splitext(fname)[0]   # e.g. F1s3
    utt_id = stem[-2:]                     # e.g. s3, s4, s5
    if utt_id not in ("s3", "s4", "s5"):
        continue
    path = os.path.join(ACE_SPEECH, fname)
    info = sf.info(path)
    # check duration after resampling to 16kHz
    dur_at_target = info.duration
    if dur_at_target < MIN_DURATION:
        continue
    speaker = stem[:-2]   # e.g. F1, M4
    speech_clips.append({"path": path, "speaker": speaker, "utt": utt_id, "stem": stem})

print(f"Usable ACE speech clips (s3/s4/s5, >={MIN_DURATION}s): {len(speech_clips)}")
VAL_CLIPS  = 5   # ~20% of training set size (26956 × 0.20 ≈ 5391)
speech_clips = random.sample(speech_clips, min(VAL_CLIPS, len(speech_clips)))
speakers = sorted(set(c["speaker"] for c in speech_clips))
print(f"Selected {len(speech_clips)} clips — speakers: {speakers}\n")

# ── Build val set ──────────────────────────────────────────────────────────────

aw_b, aw_a = design_a_weighting(TARGET_SR)

_cfs    = erb_space(FREQ_LOW, FREQ_HIGH, N_BANDS)
_fcoefs = make_erb_filters(TARGET_SR, _cfs)

import shutil
if os.path.exists(VAL_DIR):
    shutil.rmtree(VAL_DIR)
os.makedirs(VAL_DIR)

manifest_rows = []
total_chunks  = 0

for rir_rec in tqdm(ACE_RIRS, desc="ACE RIRs"):
    rir_name = f"{rir_rec['room']}_pos{rir_rec['pos']}"

    try:
        rir_signal = load_wav(rir_rec["path"])
    except Exception as e:
        print(f"  [WARN] RIR load failed {rir_rec['path']}: {e}")
        continue

    for clip in speech_clips:
        try:
            speech = load_wav(clip["path"])
        except Exception as e:
            print(f"  [WARN] speech load failed {clip['stem']}: {e}")
            continue

        reverberant = convolve_rir(speech, rir_signal)
        chunks      = chunk_utterance(reverberant, aw_b, aw_a)

        for w_idx, chunk in enumerate(chunks):
            feat = gammatone_log_energy(chunk)
            feat = spectral_whiten(feat)
            feat = (feat - global_mean) / (global_std + 1e-8)

            out_name = f"{rir_name}__{clip['stem']}__w{w_idx}.npy"
            np.save(os.path.join(VAL_DIR, out_name), feat.astype(np.float32))

            manifest_rows.append({
                "filename": f"val/{out_name}",
                "t30":      rir_rec["t30"],
                "room":     rir_rec["room"],
                "pos":      rir_rec["pos"],
                "speaker":  clip["speaker"],
                "utt":      clip["utt"],
                "shape":    f"{feat.shape[0]}x{feat.shape[1]}",
            })
            total_chunks += 1

print(f"\nDone — {total_chunks} val matrices written to {VAL_DIR}")

# ── Save manifest ──────────────────────────────────────────────────────────────

fieldnames = ["filename", "t30", "room", "pos", "speaker", "utt", "shape"]
with open(VAL_MANIFEST, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(manifest_rows)

print(f"Manifest saved → {VAL_MANIFEST}")

# ── Summary ────────────────────────────────────────────────────────────────────

from collections import defaultdict
per_room = defaultdict(int)
for r in manifest_rows:
    per_room[r["room"]] += 1

print("\nMatrices per room:")
for room in sorted(per_room):
    recs  = [r for r in manifest_rows if r["room"] == room]
    t30   = recs[0]["t30"]
    print(f"  {room:<22}  T30={t30:.3f}s  →  {per_room[room]:4d} matrices")

print(f"\nTotal val matrices : {total_chunks}")
print(f"Val manifest       : {VAL_MANIFEST}")
