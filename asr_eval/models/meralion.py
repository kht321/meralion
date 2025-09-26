"""MERaLiON model wrapper."""

from __future__ import annotations

import re
from typing import Optional

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from .base import ASRModel


PROMPT_TEMPLATE = (
    "Instruction: {q} \n"
    "Follow the text instruction based on the following audio: <SpeechHere>"
)
TRANSCRIBE_PROMPT = PROMPT_TEMPLATE.format(q="Please transcribe this speech.")


def _build_chat_prompt(tokenizer, prompt_text: str) -> str:
    conversation = [[{"role": "user", "content": prompt_text}]]
    try:
        return tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt_text


def _clean_output(text: str) -> str:
    if not text:
        return ""

    cleaned = text
    lowered = cleaned.lower()
    marker = "<speechhere>"
    last_idx = lowered.rfind(marker)
    if last_idx != -1:
        cleaned = cleaned[last_idx + len(marker) :]

    cleaned = re.sub(r"^model\s*(?:<[^>]+>)?\s*:?,?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"instruction:.*?audio:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<speechhere>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


class MERaLiON(ASRModel):
    def __init__(self, model_id: str, device, trust_remote_code: bool = True) -> None:
        self.name = model_id
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            torch_dtype=device.torch_dtype,
        ).to(str(device))
        if str(device) == "mps":
            self.model = self.model.to(dtype=torch.float16)

        tokenizer = getattr(self.processor, "tokenizer", None)
        self._prompt: Optional[str] = None
        if tokenizer is not None:
            self._prompt = _build_chat_prompt(tokenizer, TRANSCRIBE_PROMPT)

        self._model_dtype = next(self.model.parameters()).dtype

    def transcribe(self, wav, sr: int) -> str:
        inputs = {"audios": wav, "sampling_rate": int(sr)}
        if self._prompt:
            inputs["text"] = self._prompt

        features = self.processor(**inputs)
        if not isinstance(features, dict):
            features = dict(features)

        prepared = {}
        for key, value in features.items():
            if value is None:
                continue
            if hasattr(value, "to"):
                to_kwargs = {"device": str(self.device)}
                if hasattr(value, "dtype") and value.dtype.is_floating_point:
                    to_kwargs["dtype"] = self._model_dtype
                value = value.to(**to_kwargs)
            prepared[key] = value

        with torch.no_grad():
            generated = self.model.generate(**prepared, max_new_tokens=256)

        decoded = self.processor.batch_decode(generated, skip_special_tokens=True)
        transcript = decoded[0] if decoded else ""
        return _clean_output(transcript).lower()

    def close(self) -> None:
        del self.model


__all__ = ["MERaLiON"]
