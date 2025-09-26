"""Whisper model wrapper."""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from .base import ASRModel


class Whisper(ASRModel):
    def __init__(self, model_id: str, device) -> None:
        self.name = model_id
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=device.torch_dtype,
        ).to(str(device))
        if str(device) == "mps":
            self.model = self.model.to(dtype=torch.float16)

        self._forced_ids: Optional[list[list[int]]] = None
        get_prompt_ids = getattr(self.processor, "get_decoder_prompt_ids", None)
        if callable(get_prompt_ids):
            try:
                self._forced_ids = get_prompt_ids(language="en", task="transcribe")
            except Exception:
                self._forced_ids = None

    def transcribe(self, wav, sr: int) -> str:
        features = self.processor(
            audio=wav,
            sampling_rate=int(sr),
            return_tensors="pt",
        )
        features = {
            key: (value.to(device=str(self.device)) if hasattr(value, "to") else value)
            for key, value in features.items()
        }

        generation_kwargs = {"max_new_tokens": 256}
        if self._forced_ids is not None:
            generation_kwargs["forced_decoder_ids"] = self._forced_ids

        with torch.no_grad():
            generated = self.model.generate(**features, **generation_kwargs)

        decoded = self.processor.batch_decode(generated, skip_special_tokens=True)
        return (decoded[0] if decoded else "").lower()

    def close(self) -> None:
        del self.model


__all__ = ["Whisper"]
