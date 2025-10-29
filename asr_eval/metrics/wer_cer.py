"""Corpus-level WER/CER utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import jiwer

_TRANSFORM = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.SubstituteRegexes({
            r"<speaker\d+>:?\s*": "",  # Remove speaker tags like <speaker1>: or <speaker2>
            r"^model\s*:?\s*": "",      # Remove "model" prefix at start
        })
    ]
)


@dataclass
class Scores:
    wer: float
    cer: float
    n_utts: int


def corpus_scores(refs: Iterable[str], hyps: Iterable[str]) -> Scores:
    ref_list: List[str] = [
        _TRANSFORM(r or "") for r in refs
    ]
    hyp_list: List[str] = [
        _TRANSFORM(h or "") for h in hyps
    ]

    if len(ref_list) != len(hyp_list) or not ref_list:
        raise ValueError("refs and hyps must be non-empty and aligned")

    return Scores(
        wer=float(jiwer.wer(ref_list, hyp_list)),
        cer=float(jiwer.cer(ref_list, hyp_list)),
        n_utts=len(ref_list),
    )


__all__ = ["Scores", "corpus_scores"]
