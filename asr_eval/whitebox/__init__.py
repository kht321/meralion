"""
White-box evaluation utilities.

This package intentionally isolates guardrail-focused tooling so it does not
bleed into the core ASR evaluation pipeline. Import from here when building
white-box experiments.
"""

from .dataset import GuardrailSample, load_guardrail_samples

__all__ = ["GuardrailSample", "load_guardrail_samples"]
