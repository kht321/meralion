"""Helpers for building logits processors used in white-box guardrail experiments."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Set

import torch
from transformers import PreTrainedTokenizerBase
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList


class KeywordMaskingLogitsProcessor(LogitsProcessor):
    """Mask specific token ids during generation by setting their logits to ``-inf``."""

    def __init__(self, token_ids: Iterable[int]) -> None:
        self._banned: Set[int] = {int(i) for i in token_ids}
        if not self._banned:
            raise ValueError("KeywordMaskingLogitsProcessor requires at least one token id")

    @property
    def banned_token_ids(self) -> Set[int]:
        return set(self._banned)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self._banned:
            return scores
        scores[:, list(self._banned)] = -float("inf")
        return scores


def _keyword_to_token_ids(tokenizer: PreTrainedTokenizerBase, keyword: str) -> Sequence[int]:
    token_ids = tokenizer.encode(keyword, add_special_tokens=False)
    if not token_ids:
        return []
    # If the keyword spans multiple tokens, fall back to the first token so we still
    # reduce the probability of producing that span, albeit approximately.
    if len(token_ids) > 1:
        return [token_ids[0]]
    return token_ids


def build_keyword_logits_processor(
    tokenizer: Optional[PreTrainedTokenizerBase],
    rules: dict[str, Iterable[str]],
) -> Optional[LogitsProcessorList]:
    if tokenizer is None:
        return None

    banned_ids: Set[int] = set()
    for _, keywords in rules.items():
        for keyword in keywords:
            keyword = keyword.strip()
            if not keyword:
                continue
            ids = _keyword_to_token_ids(tokenizer, keyword)
            if ids:
                banned_ids.add(ids[0])

    if not banned_ids:
        return None

    processor = KeywordMaskingLogitsProcessor(banned_ids)
    return LogitsProcessorList([processor])


def extract_banned_token_strings(
    tokenizer: Optional[PreTrainedTokenizerBase],
    token_ids: Iterable[int],
) -> List[str]:
    if tokenizer is None:
        return []
    return [tokenizer.convert_ids_to_tokens(int(token_id)) for token_id in token_ids]
