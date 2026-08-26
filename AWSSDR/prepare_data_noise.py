"""Generate Stage-B (NOISY) AWSSDR training features noise_v4: convolve speech (LibriSpeech
train-clean-100) with the 855 RIRs of RIRS/ALL_RIRS2_train (571 synthetic SimShoebox + 284
real-world), add Gamper & Tashev (2018, Sec. 3.1) style noise (ambient/fan shaped-WGN + AID
transients, or synthetic babble) at 0/10/20 dB SNR, then compute the spectral decay rate
sequence per WHOLE utterance (no chunking - AWSSDR is sequence-level), saving one .npz per
utterance (DR float32 (40, N), t60, snr_db, noise_type).

This is the noisy counterpart of the CLEAN v4 set, and is paired with it deliberately:
both draw from the same 855-RIR revision of ALL_RIRS2_train and both take their targets
from T30_ground_truth.csv ("t30" column). That pairing is the whole point of this version.
The superseded noise_v3 set (26 Jul) used the *other* label file, RT30.csv, against an
earlier 866-RIR revision of the folder - and since the two CSVs disagree on 849 of the 855
shared RIRs by up to 0.126 s, a v4-vs-noise_v3 comparison confounds a label change with the
noise change. noise_v4 removes that confound: v4 -> noise_v4 differs by noise alone.

The noise machinery itself is unchanged and still copied verbatim from
CNN/prepare_data_noise.py, whose SNR mixing follows the ACE Challenge software (Eaton et al.
2016, Eqs. 2-3): noise scaled against the ITU-T P.56 Method B active speech level of the
reverberant speech. The paper (Kim & Kim 2022, Sec. III-A) used the ACE tool with six real
Aurora-4 noises; Aurora-4 is a licensed corpus we do not have, so we substitute the
validated Gamper-style synthesis (3 noise types) that already beat the DRR paper on real
noisy ACE.

K_CLEAN=34 matches prepare_data_v2.py's K_UTT_PER_RIR exactly (same SEED, same first RNG
call per RIR, same LibriSpeech pool build order), so every noisy utterance's first K_CLEAN
draws are the *exact* clean-counterpart utterances of features_awssdr_v4's u0..u33 - any
performance gap between the two sets is attributable to noise alone. N_PASSES=1 (not 3, as
the old v1_noise used against a much smaller K_CLEAN=9) because K_CLEAN=34 already reaches
855 x 34 = 29,070 utterances on its own - the same scale as v4 clean, so no extra passes
are needed to hit the paper's ~29k target.

Targets are the T30_ground_truth.csv ground truth (kept under the "T60_s"/"t60" field names
for downstream compatibility, matching prepare_data_v2.py). Utterances are capped at MAX_S
seconds to bound batch padding.

Usage: python prepare_data_noise.py [--limit N]   (--limit: only the first N RIRs, smoke)
"""

import argparse
import glob
import os
import re
import time

import librosa
import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.signal
import soundfile as sf
from scipy.signal import fftconvolve

from features import FS, MIN_SAMPLES, compute_decay_rates

HERE = os.path.dirname(os.path.abspath(__file__))                    # AWSSDR/src/
PROJECT_ROOT = os.path.dirname(HERE)                                 # AWSSDR/
PARENT_ROOT = os.path.dirname(PROJECT_ROOT)                          # T60_EST4/ (shared Speech/, Eval_data/)
RIR_DIR = os.path.join(PROJECT_ROOT, "RIRS", "Train_RIRs2")
RIR_CSV = os.path.join(PROJECT_ROOT, "RIRS", "Train_RIRs2", "T30_ground_truth.csv")
LABEL_COL = "t30"           
LIBRISPEECH_DIR = os.path.join(PARENT_ROOT, "Speech", "LibriSpeech", "train-clean-100")
NOISE_DIR = os.path.join(PARENT_ROOT, "Eval_data", "ACE_Corpus_RIRN_Single", "Single")
TRANSIENT_DIR = os.path.join(PARENT_ROOT, "Eval_data", "AID", "wavs")

VERSION = "noise_v8"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_DIR = os.path.join(DATA_DIR, "features_awssdr_" + VERSION, "train")
INDEX_CSV = os.path.join(DATA_DIR, "awssdr_train_chunks_" + VERSION + ".csv")

SEED = 42
K_CLEAN = 34                 # matches prepare_data_v2.py's K_UTT_PER_RIR (v3 clean set)
N_PASSES = 1                 # K_CLEAN alone already reaches ~29k utterances; see docstring
K_UTT_PER_RIR = K_CLEAN * N_PASSES
MAX_S = 15.0

# --- noise synthesis config (verbatim from CNN/prepare_data_noise.py) ---------------------
NOISE_ROOMS = ("Office_1", "Building_Lobby")   # ACE dev rooms only (no test-room leakage)
NOISE_TYPES = ("Ambient", "Fan", "Babble")
SNRS_DB = (0, 10, 20)
NOISE_SEG_S = 10.0           # length of the noise segment whose spectrum shapes the WGN
N_BABBLE_TALKERS = (4, 7)    # inclusive range, like the 4-7 talkers in the ACE babble
BABBLE_AMBIENT_DB = -15.0    # ambient bed level relative to babble RMS
N_TRANSIENTS = (1, 3)        # transient events per ambient/fan noise sample (inclusive)
TRANSIENT_GAIN_DB = (-5.0, 10.0)  # transient active RMS relative to the bed RMS (uniform)
TRANSIENT_MIC = "SH_MKH800"  # AID captured each event on 3 mics; use one to avoid dup events
TRANSIENT_CLASSES = ("foot_steps", "cough", "paper", "book", "keys", "pen_case",
                     "scissors", "rubber_band", "roll_stool", "zipper", "snapping",
                     "clapping")  # curated office/human subset ~ paper's "foot steps,
                                  # coughing, office equipment, etc." (59 anechoic events)
STFT_NPERSEG = 1024

MAX_SPEECH_CACHE = 1000      # babble sampling touches the whole pool; cap RAM use

_speech_cache = {}


def build_speech_pool():
    pool = []
    for spk in sorted(os.listdir(LIBRISPEECH_DIR)):
        spk_dir = os.path.join(LIBRISPEECH_DIR, spk)
        if not os.path.isdir(spk_dir):
            continue
        for chapter in sorted(os.listdir(spk_dir)):
            d = os.path.join(spk_dir, chapter)
            if not os.path.isdir(d):
                continue
            pool += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".flac")]
    return pool


def load_speech(path):
    if path not in _speech_cache:
        if len(_speech_cache) >= MAX_SPEECH_CACHE:
            _speech_cache.pop(next(iter(_speech_cache)))
        x, _ = librosa.load(path, sr=FS)
        _speech_cache[path] = x
    return _speech_cache[path]


def load_rir(path):
    r, sr = sf.read(path, always_2d=True)
    assert sr == FS, (path, sr)
    r = r[:, 0]
    return r / (np.max(np.abs(r)) + 1e-12)


def load_noise_pool():
    """ACE dev-room noise recordings, resampled to FS: {"Ambient": [...], "Fan": [...]}."""
    pool = {"Ambient": [], "Fan": []}
    for room in NOISE_ROOMS:
        for pos in ("1", "2"):
            for kind in pool:
                paths = glob.glob(os.path.join(NOISE_DIR, room, pos, "*_Noise_%s.wav" % kind))
                assert len(paths) == 1, (room, pos, kind, paths)
                pool[kind].append(librosa.load(paths[0], sr=FS)[0])
    return pool


def load_transient_pool():
    """AID anechoic transient clips (the paper's 'anechoic sound samples'), restricted to
    TRANSIENT_CLASSES on the TRANSIENT_MIC channel, silence-trimmed, with their active RMS
    precomputed. AID is 48 kHz; librosa.load(sr=FS) resamples to 16 kHz."""
    pool, per_class = [], {}
    for cls in TRANSIENT_CLASSES:
        pat = re.compile(r"^%s_\d+_%s\.wav$" % (re.escape(cls), re.escape(TRANSIENT_MIC)))
        paths = sorted(p for p in glob.glob(os.path.join(TRANSIENT_DIR, "*.wav"))
                       if pat.match(os.path.basename(p)))
        for p in paths:
            x, _ = librosa.load(p, sr=FS)
            x, _ = librosa.effects.trim(x, top_db=30)
            active = np.abs(x) > 0.05 * (np.max(np.abs(x)) + 1e-12)
            if not active.any():
                continue
            pool.append((x, float(np.sqrt(np.mean(x[active] ** 2)))))
        per_class[cls] = len(paths)
        assert paths, "no AID files for class %r mic %r" % (cls, TRANSIENT_MIC)
    print("transient pool: %d clips across %d classes %s"
          % (len(pool), len(per_class), per_class))
    return pool


def add_transients(rng, bed, transient_pool):
    """Overlay random transient events on a dry noise bed (in place)."""
    bed_rms = np.sqrt(np.mean(bed ** 2))
    for _ in range(rng.integers(N_TRANSIENTS[0], N_TRANSIENTS[1] + 1)):
        t, t_rms = transient_pool[rng.integers(len(transient_pool))]
        gain = bed_rms / (t_rms + 1e-12) * 10.0 ** (rng.uniform(*TRANSIENT_GAIN_DB) / 20.0)
        off = rng.integers(max(len(bed) - len(t), 0) + 1)
        seg = t[:len(bed) - off]
        bed[off:off + len(seg)] += gain * seg
    return bed


def shaped_wgn(rng, noise_pool, kind, n):
    """WGN of length n, spectrally shaped by a random NOISE_SEG_S segment of a `kind` recording."""
    rec = noise_pool[kind][rng.integers(len(noise_pool[kind]))]
    seg_n = min(int(NOISE_SEG_S * FS), len(rec))
    start = rng.integers(len(rec) - seg_n + 1)
    seg = rec[start:start + seg_n]
    _, _, Z = scipy.signal.stft(seg, FS, nperseg=STFT_NPERSEG)
    env = np.sqrt((np.abs(Z) ** 2).mean(axis=1))  # RMS magnitude spectrum

    wgn = rng.standard_normal(n)
    _, _, W = scipy.signal.stft(wgn, FS, nperseg=STFT_NPERSEG)
    _, out = scipy.signal.istft(W * env[:, None], FS, nperseg=STFT_NPERSEG)
    return out[:n]


def make_noise(rng, noise_pool, kind, rir, n, speech_pool, exclude_idx, transient_pool):
    """Reverberant noise signal of length n for one training mixture."""
    if kind in ("Ambient", "Fan"):
        bed = add_transients(rng, shaped_wgn(rng, noise_pool, kind, n), transient_pool)
        return fftconvolve(bed, rir)[:n]

    assert kind == "Babble", kind
    n_talkers = rng.integers(N_BABBLE_TALKERS[0], N_BABBLE_TALKERS[1] + 1)
    babble = np.zeros(n)
    for _ in range(n_talkers):
        u = rng.integers(len(speech_pool))
        while u == exclude_idx:
            u = rng.integers(len(speech_pool))
        r = fftconvolve(load_speech(speech_pool[u]), rir)
        reps = int(np.ceil(n / len(r))) + 1  # tile so any circular offset covers n samples
        off = rng.integers(len(r))
        babble += np.tile(r, reps)[off:off + n]
    amb_bed = add_transients(rng, shaped_wgn(rng, noise_pool, "Ambient", n), transient_pool)
    amb = fftconvolve(amb_bed, rir)[:n]
    babble_rms = np.sqrt(np.mean(babble ** 2))
    amb_rms = np.sqrt(np.mean(amb ** 2))
    amb *= babble_rms / (amb_rms + 1e-12) * 10.0 ** (BABBLE_AMBIENT_DB / 20.0)
    return babble + amb


def active_speech_level(x):
    """Active speech level (mean-square power) per ITU-T P.56 Method B.

    Envelope q = |x| through two cascaded 30 ms exponential smoothers; a sample is
    active for threshold c if q exceeded c within the last 200 ms (hangover). The
    active level A and threshold C satisfy A - C = 15.9 dB; log-linear interpolation
    between the bracketing geometric thresholds.
    """
    sq = float(np.sum(x ** 2))
    peak = float(np.max(np.abs(x)))
    if peak == 0.0 or sq == 0.0:
        return 0.0
    g = np.exp(-1.0 / (FS * 0.03))
    p = scipy.signal.lfilter([1 - g], [1, -g], np.abs(x))
    q = scipy.signal.lfilter([1 - g], [1, -g], p)
    hang = int(np.ceil(0.2 * FS))
    # trailing max over [k - hang, k] implements the hangover
    qmax = scipy.ndimage.maximum_filter1d(q, size=hang + 1, origin=-(hang // 2),
                                          mode="constant", cval=0.0)
    c = peak * 2.0 ** -np.arange(15, 0, -1)          # ascending thresholds
    a = len(x) - np.searchsorted(np.sort(qmax), c)   # activity count per threshold
    valid = a > 0
    c, a = c[valid], a[valid]
    A = 10 * np.log10(sq / a)                        # active level candidates [dB]
    C = 20 * np.log10(c)
    d = A - C                                        # decreasing in c
    M = 15.9
    if d[0] <= M:
        return sq / a[0]
    below = np.nonzero(d <= M)[0]
    if len(below) == 0:
        return sq / a[-1]
    j = below[0]
    frac = (d[j - 1] - M) / (d[j - 1] - d[j])
    return 10.0 ** ((A[j - 1] + frac * (A[j] - A[j - 1])) / 10.0)


def draw_utts(rng, n_pool):
    """K_UTT_PER_RIR utterance indices for one RIR: first K_CLEAN identical to the clean
    set's draw (SAME first RNG call as prepare_data_v2.py), then the rest without
    replacement from what remains. Returns a length-K_UTT_PER_RIR array."""
    clean = rng.choice(n_pool, size=K_CLEAN, replace=False)   # <- must match prepare_data_v2.py
    remaining = np.setdiff1d(np.arange(n_pool), clean)
    extra = rng.choice(remaining, size=K_UTT_PER_RIR - K_CLEAN, replace=False)
    return np.concatenate([clean, extra])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N RIRs (smoke)")
    args = ap.parse_args()

    df = pd.read_csv(RIR_CSV)
    df["basename"] = df["filename"].map(os.path.basename)
    df = df.sort_values("basename").reset_index(drop=True)
    if args.limit:
        df = df.iloc[:args.limit]

    pool = build_speech_pool()
    noise_pool = load_noise_pool()
    transient_pool = load_transient_pool()
    print("RIRs: %d  speech pool: %d files  transients: %d  K=%d  grid: %d types x %d SNRs x %d passes"
          % (len(df), len(pool), len(transient_pool), K_UTT_PER_RIR,
             len(NOISE_TYPES), len(SNRS_DB), N_PASSES))
    os.makedirs(OUT_DIR, exist_ok=True)

    t0 = time.time()
    n_files, n_capped = 0, 0
    for i, row in df.iterrows():
        stem = os.path.splitext(row["basename"])[0]
        if os.path.exists(os.path.join(OUT_DIR, "%s__u%d.npz" % (stem, K_UTT_PER_RIR - 1))):
            continue  # RIR already fully processed (skip-if-exists; delete shards to regen)
        rir = load_rir(os.path.join(RIR_DIR, row["basename"]))
        rng = np.random.default_rng([SEED, i])
        utts = draw_utts(rng, len(pool))

        for j, u in enumerate(utts):
            c = j % (len(NOISE_TYPES) * len(SNRS_DB))   # condition index within the grid
            kind = NOISE_TYPES[c // len(SNRS_DB)]
            snr = SNRS_DB[c % len(SNRS_DB)]

            speech = load_speech(pool[u])
            rev = fftconvolve(speech, rir)              # full tail kept: decay is the signal
            if len(rev) > int(MAX_S * FS):
                rev = rev[:int(MAX_S * FS)]
                n_capped += 1
            assert len(rev) >= MIN_SAMPLES, (pool[u], len(rev))

            asl = active_speech_level(rev)
            noise = make_noise(rng, noise_pool, kind, rir, len(rev), pool, u, transient_pool)
            p_n = np.mean(noise ** 2)
            gain = np.sqrt(asl / (p_n * 10.0 ** (snr / 10.0) + 1e-12))  # ACE Eq. 2-3
            mix = rev + gain * noise
            mix = mix / (np.max(np.abs(mix)) + 1e-12)

            dr = compute_decay_rates(mix)
            np.savez(os.path.join(OUT_DIR, "%s__u%d.npz" % (stem, j)),
                     DR=dr, t60=np.float32(row[LABEL_COL]),  # field kept as "t60" for downstream compat
                     snr_db=np.int16(snr), noise_type=kind)
            n_files += 1

            if i == 0 and c in (0, 3, 6):  # one debug pair per noise type at 0 dB
                sf.write(os.path.join(DATA_DIR, "debug_noisy_awssdr_%s_%ddB_%s.wav"
                                      % (kind, snr, VERSION)), mix, FS)
                sf.write(os.path.join(DATA_DIR, "debug_noise_awssdr_%s_%s.wav" % (kind, VERSION)),
                         noise / (np.max(np.abs(noise)) + 1e-12), FS)

        if (i + 1) % 20 == 0 or i == len(df) - 1:
            el = time.time() - t0
            print("[%d/%d] utterances so far: %d (%d capped at %.0f s)  (%.1f s elapsed, %.2f s/RIR)"
                  % (i + 1, len(df), n_files, n_capped, MAX_S, el, el / (i + 1)), flush=True)

    rows = []
    for f in sorted(os.listdir(OUT_DIR)):
        with np.load(os.path.join(OUT_DIR, f)) as z:
            rows.append((f, float(z["t60"]), z["DR"].shape[1],
                         str(z["noise_type"]), int(z["snr_db"])))
    idx = pd.DataFrame(rows, columns=["filename", "T60_s", "n_frames", "noise_type", "snr_db"])
    idx.to_csv(INDEX_CSV, index=False)
    print("utterances: %d  total SDR frames: %d  -> %s"
          % (len(idx), idx["n_frames"].sum(), INDEX_CSV))


if __name__ == "__main__":
    main()
