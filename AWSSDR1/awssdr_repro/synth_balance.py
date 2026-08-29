"""Duration-balanced AWSSDR training-set synthesis (revision 2).

This is intentionally separate from :mod:`synth_train` so the completed
EARS+TIMIT-short dataset remains exactly reproducible.  Each RIR produces 34
examples: 14 exact-length TIMIT word-sequence crops and 20 EARS examples, of
which one or two are medium-duration speech-active crops.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import fftconvolve

from . import config as cfg
from .dataset_io import prepare_feature_dir, safe_stem, save_sample, write_index
from .ears import SpeechClip, build_ears_manifest, group_by_speaker, select_balanced_clips
from .features import compute_decay_rates, load_audio_16k
from .labels import load_train_t30_labels
from .noise import load_noise_pool, load_rir, load_transient_pool, make_noise, mix_at_snr


DEFAULT_OUT = cfg.GENERATED_DIR / "data-train-t30-ears-timit-duration-balanced-noisy"
CONDITIONS = tuple(
    (noise_type, snr_db)
    for noise_type in cfg.NOISE_TYPES
    for snr_db in cfg.SNRS_DB
)
GROUP_SPECS = (
    ("timit_60_79", "timit", 5, 60, 79),
    ("timit_80_119", "timit", 5, 80, 119),
    ("timit_120_199", "timit", 4, 120, 199),
)
EARS_MEDIUM_MIN_FRAMES = 280
EARS_MEDIUM_MAX_FRAMES = 360


@dataclass(frozen=True)
class Word:
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class TimitUtterance:
    clip: SpeechClip
    words: tuple[Word, ...]


@dataclass
class Plan:
    source: str
    duration_group: str
    requested_decay_frames: int | None
    condition: tuple[str, int] | None = None
    clip: SpeechClip | None = None


def samples_for_decay_frames(n_decay_frames: int) -> int:
    """Minimum waveform samples that yield exactly ``n_decay_frames``."""
    if n_decay_frames < 1:
        raise ValueError("n_decay_frames must be positive")
    n_lmfe_frames = cfg.FIT_LEN + cfg.FIT_HOP * (n_decay_frames - 1)
    return cfg.N_FFT + cfg.HOP_LENGTH * (n_lmfe_frames - 1)


def _read_words(path: Path) -> tuple[Word, ...]:
    words = []
    with path.open() as handle:
        for line in handle:
            start, end, text = line.rstrip().split(maxsplit=2)
            words.append(Word(int(start), int(end), text))
    if not words:
        raise ValueError(f"empty TIMIT word alignment: {path}")
    return tuple(words)


def build_timit_word_manifest(root: Path = cfg.TIMIT_DIR) -> list[TimitUtterance]:
    """Load every SI/SX utterance; duration filtering belongs to crop selection."""
    utterances = []
    for path in sorted(root.glob("dr*-*/*.wav")):
        sentence_type = path.stem[:2].lower()
        if sentence_type not in {"si", "sx"}:
            continue
        wrd_path = path.with_suffix(".wrd")
        if not wrd_path.exists():
            raise FileNotFoundError(wrd_path)
        speaker_dir = path.parent.name
        dialect, speaker_code = speaker_dir.split("-", 1)
        clip = SpeechClip(
            path=path,
            speaker=f"timit:{speaker_dir}",
            category=f"timit_{sentence_type}",
            source="timit",
            duration_s=float(sf.info(path).duration),
            gender="Female" if speaker_code.startswith("f") else "Male",
            dialect=dialect,
        )
        if sf.info(path).samplerate != cfg.FS:
            raise ValueError(f"TIMIT word alignments require {cfg.FS} Hz audio: {path}")
        utterances.append(TimitUtterance(clip, _read_words(wrd_path)))
    speakers = {u.clip.speaker for u in utterances}
    if len(speakers) != 16:
        raise ValueError(f"expected 16 TIMIT speakers, found {len(speakers)}")
    return utterances


def _word_candidates(
    utterance: TimitUtterance, target_n: int, rir_n: int
) -> list[tuple[float, int, int]]:
    """Return complete consecutive-word spans capable of reaching target length."""
    preferred_tail = 0.15 * cfg.FS
    candidates = []
    for first in range(len(utterance.words)):
        for last in range(first, len(utterance.words)):
            dry_n = utterance.words[last].end - utterance.words[first].start
            tail_n = target_n - dry_n
            if tail_n < 0:
                break
            if tail_n > rir_n - 1:
                continue
            # Prefer the intended 50--250 ms retained tail, but permit longer
            # tails when short target bands leave no exact complete-word option.
            outside = max(0.05 * cfg.FS - tail_n, 0, tail_n - 0.25 * cfg.FS)
            score = outside * 10.0 + abs(tail_n - preferred_tail)
            candidates.append((float(score), first, last))
    return candidates


def select_timit_word_crop(
    by_speaker: dict[str, list[TimitUtterance]],
    speaker: str,
    target_n: int,
    rir_n: int,
    rng: np.random.Generator,
) -> tuple[TimitUtterance, int, int]:
    """Choose an exact-length-capable word sequence from the requested speaker."""
    candidates = []
    for utterance in by_speaker[speaker]:
        for score, first, last in _word_candidates(utterance, target_n, rir_n):
            candidates.append((score, utterance, first, last))
    if not candidates:
        raise ValueError(
            f"no complete-word TIMIT crop for speaker={speaker}, target_n={target_n}, rir_n={rir_n}"
        )
    candidates.sort(key=lambda item: item[0])
    best_score = candidates[0][0]
    near_best = [item for item in candidates if item[0] <= best_score + 0.025 * cfg.FS]
    _, utterance, first, last = near_best[int(rng.integers(len(near_best)))]
    return utterance, first, last


def _active_crop(x: np.ndarray, dry_n: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Select a high-activity fixed-length window without changing its samples."""
    if len(x) < dry_n:
        raise ValueError(f"EARS clip too short for crop: {len(x)} < {dry_n}")
    if len(x) == dry_n:
        return x, 0
    max_start = len(x) - dry_n
    random_starts = rng.integers(max_start + 1, size=24)
    grid_starts = np.linspace(0, max_start, 12, dtype=int)
    starts = np.unique(np.concatenate([random_starts, grid_starts]))
    # The lower-quartile 20 ms frame energy rewards windows that stay active,
    # rather than windows containing one isolated loud event.
    frame_n = max(1, round(0.020 * cfg.FS))
    scores = []
    for start in starts:
        crop = x[start : start + dry_n]
        usable = len(crop) // frame_n * frame_n
        frame_rms = np.sqrt(np.mean(crop[:usable].reshape(-1, frame_n) ** 2, axis=1))
        scores.append(float(np.quantile(frame_rms, 0.25)))
    start = int(starts[int(np.argmax(scores))])
    return x[start : start + dry_n], start


def _make_plans(rir_index: int, rng: np.random.Generator) -> list[Plan]:
    plans = []
    for group, source, count, low, high in GROUP_SPECS:
        for _ in range(count):
            plans.append(Plan(source, group, int(rng.integers(low, high + 1))))
    n_medium = 1 + (rir_index % 2)
    for _ in range(n_medium):
        frames = int(rng.integers(EARS_MEDIUM_MIN_FRAMES, EARS_MEDIUM_MAX_FRAMES + 1))
        plans.append(Plan("ears", "ears_280_360", frames))
    for _ in range(cfg.N_EARS_PER_RIR - n_medium):
        plans.append(Plan("ears", "ears_full", None))

    # Rotate each duration group independently through all 9 noise/SNR pairs.
    group_positions: dict[str, int] = {}
    group_offsets = {name: i * 2 for i, name in enumerate(
        ["timit_60_79", "timit_80_119", "timit_120_199", "ears_280_360", "ears_full"]
    )}
    fixed_counts = {name: count for name, _, count, _, _ in GROUP_SPECS}

    def previous_group_samples(group: str) -> int:
        if group in fixed_counts:
            return rir_index * fixed_counts[group]
        pairs, remainder = divmod(rir_index, 2)
        if group == "ears_280_360":
            return pairs * 3 + remainder
        if group == "ears_full":
            return pairs * 37 + remainder * 19
        raise ValueError(group)

    for plan in plans:
        position = group_positions.get(plan.duration_group, 0)
        global_position = previous_group_samples(plan.duration_group) + position
        condition_index = (group_offsets[plan.duration_group] + global_position) % len(CONDITIONS)
        plan.condition = CONDITIONS[condition_index]
        group_positions[plan.duration_group] = position + 1
    rng.shuffle(plans)
    return plans


def _assign_clips(
    plans: list[Plan],
    ears_by_speaker: dict[str, list[SpeechClip]],
    timit_by_speaker: dict[str, list[TimitUtterance]],
    rir_n: int,
    rng: np.random.Generator,
) -> None:
    ears = select_balanced_clips(ears_by_speaker, rng, cfg.N_EARS_PER_RIR)
    rng.shuffle(ears)
    # Assign the few medium plans first so a selected short EARS file cannot
    # make an otherwise valid exact crop fail. Remaining clips stay full length.
    for plan in (p for p in plans if p.duration_group == "ears_280_360"):
        target_n = samples_for_decay_frames(plan.requested_decay_frames)
        max_tail = min(round(0.25 * cfg.FS), rir_n - 1)
        eligible = [i for i, clip in enumerate(ears) if clip.duration_s * cfg.FS >= target_n - max_tail]
        if not eligible:
            raise ValueError(f"no selected EARS clip long enough for {plan.requested_decay_frames} frames")
        plan.clip = ears.pop(eligible[int(rng.integers(len(eligible)))])

    ears_iter = iter(ears)
    for plan in plans:
        if plan.duration_group == "ears_full":
            plan.clip = next(ears_iter)

    # Match the 14 TIMIT requests to 14 distinct speakers that can produce a
    # complete-word crop at that exact length for this RIR. A simple random
    # speaker draw fails for the occasional very short RIR.
    timit_plans = [plan for plan in plans if plan.source == "timit"]
    speakers = list(timit_by_speaker)
    edges: dict[int, list[str]] = {}
    for plan_index, plan in enumerate(timit_plans):
        target_n = samples_for_decay_frames(plan.requested_decay_frames)
        eligible = []
        for speaker in speakers:
            if any(
                _word_candidates(utterance, target_n, rir_n)
                for utterance in timit_by_speaker[speaker]
            ):
                eligible.append(speaker)
        edges[plan_index] = list(rng.permutation(eligible))

    speaker_to_plan: dict[str, int] = {}

    def augment(plan_index: int, seen: set[str]) -> bool:
        for speaker in edges[plan_index]:
            if speaker in seen:
                continue
            seen.add(speaker)
            previous = speaker_to_plan.get(speaker)
            if previous is None or augment(previous, seen):
                speaker_to_plan[speaker] = plan_index
                return True
        return False

    for plan_index in sorted(edges, key=lambda index: len(edges[index])):
        if not augment(plan_index, set()):
            raise ValueError(
                f"cannot assign 14 distinct feasible TIMIT speakers for rir_n={rir_n}"
            )
    for speaker, plan_index in speaker_to_plan.items():
        timit_plans[plan_index].clip = timit_by_speaker[speaker][0].clip


def _synth_timit(
    plan: Plan,
    timit_by_speaker: dict[str, list[TimitUtterance]],
    rir: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, SpeechClip, dict]:
    target_n = samples_for_decay_frames(plan.requested_decay_frames)
    utterance, first, last = select_timit_word_crop(
        timit_by_speaker, plan.clip.speaker, target_n, len(rir), rng
    )
    x = load_audio_16k(utterance.clip.path)
    first_word, last_word = utterance.words[first], utterance.words[last]
    dry = x[first_word.start : last_word.end]
    reverberant = fftconvolve(dry, rir)[:target_n]
    if len(reverberant) != target_n:
        raise RuntimeError("TIMIT crop cannot reach requested exact output length")
    return reverberant, utterance.clip, {
        "dry_crop_start_sample": first_word.start,
        "dry_crop_end_sample": last_word.end,
        "word_start": first,
        "word_end": last,
        "word_start_text": first_word.text,
        "word_end_text": last_word.text,
        "dry_crop_duration_s": len(dry) / cfg.FS,
    }


def _synth_ears(
    plan: Plan, rir: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, SpeechClip, dict]:
    x = load_audio_16k(plan.clip.path)
    if plan.duration_group == "ears_full":
        dry, start = x, 0
        reverberant = fftconvolve(dry, rir)
    else:
        target_n = samples_for_decay_frames(plan.requested_decay_frames)
        tail_max = min(round(0.25 * cfg.FS), len(rir) - 1)
        tail_min = min(round(0.05 * cfg.FS), tail_max)
        tail_n = int(rng.integers(tail_min, tail_max + 1))
        dry_n = target_n - tail_n
        dry, start = _active_crop(x, dry_n, rng)
        reverberant = fftconvolve(dry, rir)[:target_n]
        if len(reverberant) != target_n:
            raise RuntimeError("EARS crop cannot reach requested exact output length")
    return reverberant, plan.clip, {
        "dry_crop_start_sample": start,
        "dry_crop_end_sample": start + len(dry),
        "word_start": "",
        "word_end": "",
        "word_start_text": "",
        "word_end_text": "",
        "dry_crop_duration_s": len(dry) / cfg.FS,
    }


def build(out_dir: Path = DEFAULT_OUT, limit_rirs: int = 0) -> pd.DataFrame:
    labels = load_train_t30_labels()
    rir_paths = sorted(p for p in cfg.TRAIN_RIR_DIR.glob("*.wav") if p.name in labels)
    if limit_rirs:
        rir_paths = rir_paths[:limit_rirs]
    if not rir_paths:
        raise FileNotFoundError("no labelled RIR wavs found")

    ears_pool = build_ears_manifest()
    ears_by_speaker = group_by_speaker(ears_pool)
    timit_pool = build_timit_word_manifest()
    timit_by_speaker: dict[str, list[TimitUtterance]] = {}
    for utterance in timit_pool:
        timit_by_speaker.setdefault(utterance.clip.speaker, []).append(utterance)
    noise_pool = load_noise_pool()
    transient_pool = load_transient_pool()
    prepare_feature_dir(out_dir)

    rows = []
    start_time = time.time()
    for rir_index, rir_path in enumerate(rir_paths):
        rir = load_rir(rir_path)
        rng = np.random.default_rng([cfg.SEED, 2, rir_index])
        plans = _make_plans(rir_index, rng)
        _assign_clips(plans, ears_by_speaker, timit_by_speaker, len(rir), rng)
        target_t30 = labels[rir_path.name]
        rir_id = safe_stem(rir_path.stem)

        for utt_index, plan in enumerate(plans):
            if plan.source == "timit":
                reverberant, clip, crop_meta = _synth_timit(plan, timit_by_speaker, rir, rng)
            else:
                reverberant, clip, crop_meta = _synth_ears(plan, rir, rng)
            noise_type, snr_db = plan.condition
            noise = make_noise(
                rng, noise_pool, noise_type, rir, len(reverberant),
                ears_pool, clip, transient_pool,
            )
            mixture = mix_at_snr(reverberant, noise, snr_db)
            dr = compute_decay_rates(mixture)
            if plan.requested_decay_frames is not None and dr.shape[1] != plan.requested_decay_frames:
                raise RuntimeError(
                    f"requested {plan.requested_decay_frames} frames, got {dr.shape[1]}"
                )
            sample_id = (
                f"train2__rir_{rir_id}__u{utt_index:02d}__{plan.duration_group}__"
                f"{noise_type}__{snr_db}dB"
            )
            save_sample(out_dir, sample_id, dr, target_t30)
            rows.append({
                "sample_id": sample_id,
                "split": "train",
                "rir_index": rir_index,
                "rir_filename": rir_path.name,
                "target_t30": target_t30,
                "duration_group": plan.duration_group,
                "requested_decay_frames": plan.requested_decay_frames,
                "actual_decay_frames": dr.shape[1],
                "speech_path": str(clip.path),
                "speech_source": clip.source,
                "speech_speaker": clip.speaker,
                "speech_category": clip.category,
                "speech_gender": clip.gender,
                "speech_dialect": clip.dialect,
                **crop_meta,
                "output_duration_s": len(mixture) / cfg.FS,
                "noise_type": noise_type,
                "snr_db": snr_db,
                "n_samples": len(mixture),
                "n_decay_frames": dr.shape[1],
            })

        if (rir_index + 1) % 10 == 0 or rir_index == len(rir_paths) - 1:
            print(
                f"[{rir_index + 1}/{len(rir_paths)}] samples={len(rows)} "
                f"elapsed={time.time() - start_time:.1f}s"
            )

    write_index(out_dir, rows)
    print(f"wrote {len(rows)} samples to {out_dir}")
    return pd.DataFrame(rows)


def preflight(limit_rirs: int = 0) -> None:
    """Check all deterministic crop assignments without computing audio features."""
    labels = load_train_t30_labels()
    rir_paths = sorted(p for p in cfg.TRAIN_RIR_DIR.glob("*.wav") if p.name in labels)
    if limit_rirs:
        rir_paths = rir_paths[:limit_rirs]
    ears_by_speaker = group_by_speaker(build_ears_manifest())
    timit_by_speaker: dict[str, list[TimitUtterance]] = {}
    for utterance in build_timit_word_manifest():
        timit_by_speaker.setdefault(utterance.clip.speaker, []).append(utterance)

    checked = 0
    for rir_index, rir_path in enumerate(rir_paths):
        info = sf.info(rir_path)
        rir_n = round(info.frames * cfg.FS / info.samplerate)
        rng = np.random.default_rng([cfg.SEED, 2, rir_index])
        plans = _make_plans(rir_index, rng)
        _assign_clips(plans, ears_by_speaker, timit_by_speaker, rir_n, rng)
        for plan in plans:
            if plan.source == "timit":
                target_n = samples_for_decay_frames(plan.requested_decay_frames)
                select_timit_word_crop(
                    timit_by_speaker, plan.clip.speaker, target_n, rir_n, rng
                )
            checked += 1
    print(f"preflight PASS: {len(rir_paths)} RIRs, {checked} deterministic assignments")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit-rirs", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        preflight(args.limit_rirs)
    else:
        build(args.out, args.limit_rirs)


if __name__ == "__main__":
    main()
