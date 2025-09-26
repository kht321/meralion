"""Base interfaces for ASR models."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ASRModel(ABC):
    """Contract for ASR model wrappers."""

    name: str

    @abstractmethod
    def transcribe(self, wav, sr: int) -> str:
        """Return a transcript for the provided waveform."""

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the model."""


__all__ = ["ASRModel"]
