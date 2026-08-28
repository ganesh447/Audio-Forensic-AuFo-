from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from . import config as cfg


@dataclass(frozen=True)
class SpeechClip:
    path: Path
    speaker: str
    category: str


def is_allowed_ears_file(path: Path) -> bool:
    name = path.name
    return (
        name.startswith("rainbow_")
        or name.startswith("sentences_")
        or name.startswith("freeform_speech_")
        or (name.startswith("emo_") and name.endswith("_sentences.wav"))
    )


def category_for(path: Path) -> str:
    name = path.name
    if name.startswith("freeform_speech_"):
        return "freeform_speech"
    if name.startswith("emo_"):
        return "emo_sentences"
    return name.split("_", 1)[0]


def build_ears_manifest(root: Path = cfg.EARS_DIR) -> list[SpeechClip]:
    clips = []
    for path in sorted(root.rglob("*.wav")):
        if is_allowed_ears_file(path):
            clips.append(
                SpeechClip(path=path, speaker=path.parent.name, category=category_for(path))
            )
    if not clips:
        raise FileNotFoundError(f"no allowed EARS wav files found under {root}")
    speakers = {c.speaker for c in clips}
    if len(speakers) != 10:
        raise ValueError(f"expected 10 EARS speakers, found {len(speakers)}: {speakers}")
    return clips


def group_by_speaker(clips: Iterable[SpeechClip]) -> dict[str, list[SpeechClip]]:
    grouped: dict[str, list[SpeechClip]] = {}
    for clip in clips:
        grouped.setdefault(clip.speaker, []).append(clip)
    return {k: sorted(v, key=lambda c: str(c.path)) for k, v in sorted(grouped.items())}


def select_balanced_clips(
    clips_by_speaker: dict[str, list[SpeechClip]],
    rng: np.random.Generator,
    k: int,
) -> list[SpeechClip]:
    speakers = list(clips_by_speaker)
    speaker_order = list(rng.permutation(speakers))
    pools = {
        spk: list(rng.permutation(clips_by_speaker[spk]))
        for spk in speakers
    }
    positions = {spk: 0 for spk in speakers}
    selected = []
    for i in range(k):
        spk = speaker_order[i % len(speaker_order)]
        if positions[spk] >= len(pools[spk]):
            pools[spk] = list(rng.permutation(clips_by_speaker[spk]))
            positions[spk] = 0
        selected.append(pools[spk][positions[spk]])
        positions[spk] += 1
    return selected

