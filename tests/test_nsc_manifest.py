from __future__ import annotations

from pathlib import Path

from asr_eval.datasets.nsc_manifest import build_part1_manifest


def test_build_part1_manifest_smoke():
    root = Path("data/NSC")
    rows = build_part1_manifest(root)
    assert rows, "expected at least one NSC entry"
    sample = rows[0]
    assert sample["audio_path"].endswith(".WAV")
    assert sample["text"], "text should not be empty"
