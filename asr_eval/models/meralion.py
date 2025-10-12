"""MERaLiON model wrapper."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, PreTrainedTokenizerBase
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from .base import ASRModel


PROMPT_TEMPLATE = (
    "Instruction: {q} \n"
    "Follow the text instruction based on the following audio: <SpeechHere>"
)
TRANSCRIBE_PROMPT = PROMPT_TEMPLATE.format(q="Please transcribe this speech.")

# Minimal keyword rules for white-box guardrail experiments (opt-in).
DEFAULT_GUARDRAIL_RULES: Dict[str, Tuple[str, ...]] = {
    "profanity": ("fuck", "piss", "asshole"),
    "hate_speech": ("shemale", "tranny", "slut"),
    "violence": ("rape", "terrorist"),
}


@dataclass(frozen=True)
class GuardrailResult:
    cleaned: str
    rule_hits: List[Dict[str, str]]


def _build_chat_prompt(tokenizer: PreTrainedTokenizerBase, prompt_text: str) -> str:
    conversation = [[{"role": "user", "content": prompt_text}]]
    try:
        return tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt_text


def _normalize_rules(rules: Optional[Dict[str, Iterable[str]]]) -> Dict[str, Tuple[str, ...]]:
    if not rules:
        return {category: tuple(items) for category, items in DEFAULT_GUARDRAIL_RULES.items()}

    normalized: Dict[str, Tuple[str, ...]] = {}
    for category, keywords in rules.items():
        lowered = tuple(sorted({kw.strip().lower() for kw in keywords if kw.strip()}))
        if lowered:
            normalized[category] = lowered
    return normalized or {category: tuple(items) for category, items in DEFAULT_GUARDRAIL_RULES.items()}


def _clean_output(text: str, rules: Optional[Dict[str, Tuple[str, ...]]] = None) -> GuardrailResult:
    if not text:
        return GuardrailResult(cleaned="", rule_hits=[])

    cleaned = text
    lowered = cleaned.lower()
    marker = "<speechhere>"
    last_idx = lowered.rfind(marker)
    if last_idx != -1:
        cleaned = cleaned[last_idx + len(marker) :]

    cleaned = re.sub(r"^model\s*(?:<[^>]+>)?\s*:?,?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"instruction:.*?audio:\s*", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<speechhere>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip()

    rule_hits: List[Dict[str, str]] = []
    if rules:
        lowered_clean = cleaned.lower()
        seen = set()
        for category, keywords in rules.items():
            for keyword in keywords:
                pattern = rf"\b{re.escape(keyword)}\b"
                if re.search(pattern, lowered_clean):
                    hit_id = (category, keyword)
                    if hit_id in seen:
                        continue
                    seen.add(hit_id)
                    rule_hits.append({"category": category, "keyword": keyword})
                    cleaned = re.sub(pattern, "[BLOCKED]", cleaned, flags=re.IGNORECASE)

    return GuardrailResult(cleaned=cleaned, rule_hits=rule_hits)


def _topk_decoder_trace(
    scores: Sequence[torch.Tensor],
    tokenizer: Optional[PreTrainedTokenizerBase],
    *,
    top_k: int = 5,
) -> List[List[Dict[str, float]]]:
    if tokenizer is None:
        return []

    trace: List[List[Dict[str, float]]] = []
    for step_scores in scores:
        if step_scores.ndim < 2:
            continue
        probs = step_scores.detach().softmax(dim=-1)
        top_values, top_indices = torch.topk(probs, k=min(top_k, probs.shape[-1]), dim=-1)
        step_trace: List[Dict[str, float]] = []
        for idx_tensor, value_tensor in zip(top_indices[0], top_values[0]):
            token = tokenizer.convert_ids_to_tokens(int(idx_tensor))
            step_trace.append({"token": token, "prob": float(value_tensor)})
        trace.append(step_trace)
    return trace


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

        tokenizer: Optional[PreTrainedTokenizerBase] = getattr(self.processor, "tokenizer", None)
        self._prompt: Optional[str] = None
        self._tokenizer = tokenizer
        if tokenizer is not None:
            self._prompt = _build_chat_prompt(tokenizer, TRANSCRIBE_PROMPT)

        self._model_dtype = next(self.model.parameters()).dtype
        self._guardrail_enabled = False
        self._guardrail_rules = _normalize_rules(None)
        self._capture_decoder_traces = False
        self._logits_processor: Optional[LogitsProcessorList] = None

    def set_guardrail_rules(self, rules: Optional[Dict[str, Iterable[str]]]) -> None:
        self._guardrail_rules = _normalize_rules(rules)

    def enable_guardrail(self, rules: Optional[Dict[str, Iterable[str]]] = None) -> None:
        if rules is not None:
            self.set_guardrail_rules(rules)
        self._guardrail_enabled = True

    def disable_guardrail(self) -> None:
        self._guardrail_enabled = False

    def set_capture_decoder_traces(self, enabled: bool) -> None:
        self._capture_decoder_traces = bool(enabled)

    def set_logits_processor(
        self,
        processor: Optional[
            LogitsProcessor
            | LogitsProcessorList
            | Iterable[LogitsProcessor]
        ],
    ) -> None:
        if processor is None:
            self._logits_processor = None
            return
        if isinstance(processor, LogitsProcessorList):
            self._logits_processor = processor
        elif isinstance(processor, LogitsProcessor):
            self._logits_processor = LogitsProcessorList([processor])
        else:
            proc_list = LogitsProcessorList(list(processor))
            self._logits_processor = proc_list if len(proc_list) else None

    @property
    def tokenizer(self) -> Optional[PreTrainedTokenizerBase]:
        return self._tokenizer

    def transcribe(self, wav, sr: int, *, return_metadata: bool = False, use_logits_masking: bool = False):
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

        # Build logits processor if masking is enabled
        active_logits_processor = None
        banned_token_info = None
        if use_logits_masking and self._guardrail_enabled and self._tokenizer:
            from asr_eval.whitebox.logits import build_keyword_logits_processor, extract_banned_token_strings
            active_logits_processor = build_keyword_logits_processor(
                self._tokenizer,
                self._guardrail_rules
            )
            if active_logits_processor:
                banned_ids = list(active_logits_processor[0].banned_token_ids)
                banned_token_info = {
                    "banned_token_ids": banned_ids,
                    "banned_tokens": extract_banned_token_strings(self._tokenizer, banned_ids)
                }

        with torch.no_grad():
            gen_kwargs = dict(max_new_tokens=128, do_sample=False, num_beams=1)
            capture_traces = self._capture_decoder_traces
            if capture_traces:
                gen_kwargs.update(output_scores=True, return_dict_in_generate=True)

            # Use active_logits_processor if masking enabled, otherwise use stored processor
            if active_logits_processor is not None:
                gen_kwargs["logits_processor"] = active_logits_processor
            elif self._logits_processor is not None:
                gen_kwargs["logits_processor"] = self._logits_processor

            generated = self.model.generate(
                **prepared,
                **gen_kwargs,
            )

        scores = None
        if capture_traces:
            sequences = generated.sequences
            scores = generated.scores
        else:
            sequences = generated

        decoded = self.processor.batch_decode(sequences, skip_special_tokens=True)
        transcript = decoded[0] if decoded else ""
        guardrail_rules = self._guardrail_rules if self._guardrail_enabled else None
        guardrail_result = _clean_output(transcript, guardrail_rules)
        final_text = guardrail_result.cleaned.lower()

        metadata = {
            "raw": transcript,
            "cleaned": guardrail_result.cleaned,
            "final_text": final_text,
            "rule_hits": guardrail_result.rule_hits,
            "guardrail_enabled": self._guardrail_enabled,
            "logits_masking_enabled": use_logits_masking and active_logits_processor is not None,
            "logits_processor_active": self._logits_processor is not None or active_logits_processor is not None,
        }

        if banned_token_info:
            metadata.update(banned_token_info)

        if capture_traces and scores is not None:
            metadata["decoder_trace"] = _topk_decoder_trace(scores, self._tokenizer)

        if return_metadata:
            return metadata
        return final_text

    def close(self) -> None:
        del self.model


__all__ = ["MERaLiON", "DEFAULT_GUARDRAIL_RULES"]
