"""Utilities to build NSC manifests for robustness/fairness experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional


def _iter_wav_files(part_dir: Path) -> Iterator[Path]:
    for wav_path in sorted(part_dir.rglob("*.WAV")):
        if wav_path.name.startswith("._"):
            continue
        yield wav_path


def _parse_part1_transcripts(root: Path) -> Dict[str, str]:
    transcripts: Dict[str, str] = {}
    for txt_path in root.glob("2000*.TXT"):
        with txt_path.open("r", encoding="utf-8-sig") as handle:
            current_id: Optional[str] = None
            for raw_line in handle:
                line = raw_line.rstrip().replace("\u3000", " ")
                if not line:
                    continue
                if line[0].isdigit():
                    parts = line.split("\t", 1)
                    current_id = parts[0].strip()
                    if len(parts) > 1:
                        transcripts[current_id] = parts[1].strip()
                    else:
                        transcripts[current_id] = ""
                elif line.startswith("\t") and current_id is not None:
                    transcripts[current_id] = line.strip()
                    current_id = None
                else:
                    current_id = None
    return transcripts


def build_part1_manifest(root: Path) -> List[dict]:
    part_dir = root / "Part 1 - local accents"
    if not part_dir.exists():
        raise FileNotFoundError(f"Missing NSC part 1 directory: {part_dir}")

    transcripts = _parse_part1_transcripts(part_dir)
    rows: List[dict] = []

    for wav_path in _iter_wav_files(part_dir):
        stem = wav_path.stem
        text = transcripts.get(stem)
        if text is None:
            continue
        rel_path = wav_path.relative_to(root)
        speaker_id = wav_path.parents[1].name
        session_id = wav_path.parent.name
        rows.append(
            {
                "audio": {"path": str(rel_path)},
                "audio_path": str(rel_path),
                "utt_id": stem,
                "text": text,
                "speaker": speaker_id,
                "session": session_id,
                "part": "part1_local_accents",
            }
        )
    return rows


def write_manifest(rows: Iterable[dict], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build NSC JSONL manifest (part 1)")
    parser.add_argument(
        "--nsc-root",
        type=Path,
        default=Path("data/NSC"),
        help="Root directory of the NSC dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/nsc_part1.jsonl"),
        help="Destination JSONL manifest path",
    )
    args = parser.parse_args()

    rows = build_part1_manifest(args.nsc_root)
    write_manifest(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
