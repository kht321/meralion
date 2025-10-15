"""
White-box evaluation utilities.

This package intentionally isolates guardrail-focused tooling so it does not
bleed into the core ASR evaluation pipeline. Import from here when building
white-box experiments.
"""

from .dataset import GuardrailSample, load_guardrail_samples
from .logits import (
    KeywordMaskingLogitsProcessor,
    build_keyword_logits_processor,
    extract_banned_token_strings,
)
from .run_guardrail_eval import run_guardrail_evaluation

__all__ = [
    "GuardrailSample",
    "KeywordMaskingLogitsProcessor",
    "build_keyword_logits_processor",
    "extract_banned_token_strings",
    "load_guardrail_samples",
    "run_guardrail_evaluation",
]
