from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

DEFAULT_AUDIO_ROOT = Path("data/guardrails/audio")
DEFAULT_TRANSCRIPTS_PATH = Path("data/guardrails/transcripts.json")


@dataclass(frozen=True)
class GuardrailSample:
    """Single audio sample used in the white-box guardrail evaluation."""

    category: str
    audio_path: Path
    transcript: str


def load_guardrail_samples(
    audio_root: Path | str = DEFAULT_AUDIO_ROOT,
    transcripts_path: Path | str = DEFAULT_TRANSCRIPTS_PATH,
    *,
    strict: bool = True,
) -> List[GuardrailSample]:
    """
    Load guardrail evaluation samples.

    Parameters
    ----------
    audio_root:
        Root directory that contains the per-category audio folders
        (for example, ``data/guardrails/audio``).
    transcripts_path:
        JSON file describing the samples. Expected schema:
        ``[{"category": "...", "file": "...", "transcript": "..."}]``.
    strict:
        When True, raise ``FileNotFoundError`` if any referenced audio clip is missing.

    Returns
    -------
    list[GuardrailSample]
        Guardrail samples with resolved audio paths and transcripts.
    """

    audio_root = Path(audio_root)
    transcripts_path = Path(transcripts_path)

    with transcripts_path.open("r", encoding="utf-8") as fh:
        try:
            raw_records: Sequence[dict] = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {transcripts_path}") from exc

    samples: List[GuardrailSample] = []
    for idx, record in enumerate(raw_records):
        try:
            category = record["category"]
            filename = record["file"]
            transcript = record["transcript"]
        except KeyError as exc:
            raise ValueError(
                f"Missing key {exc!s} in record #{idx} of {transcripts_path}"
            ) from exc

        audio_path = audio_root / category / filename
        if strict and not audio_path.exists():
            raise FileNotFoundError(f"Audio clip not found: {audio_path}")

        samples.append(
            GuardrailSample(
                category=category,
                audio_path=audio_path,
                transcript=transcript,
            )
        )

    return samples


def iter_audio_paths(samples: Iterable[GuardrailSample]) -> Iterable[Path]:
    """Yield audio paths from a sequence of ``GuardrailSample`` instances."""

    for sample in samples:
        yield sample.audio_path
