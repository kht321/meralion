"""Run white-box guardrail evaluation for MERaLiON models."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Sequence

from asr_eval.audio import TARGET_SR, load_audio, resample_np
from asr_eval.device import Device
from asr_eval.models.meralion import DEFAULT_GUARDRAIL_RULES, MERaLiON
from asr_eval.whitebox import GuardrailSample, load_guardrail_samples

HARMFUL_CATEGORIES = {"profanity", "hate_speech", "violence", "pii"}


@dataclass(frozen=True)
class RunResult:
    records: List[dict]
    mean_latency_ms: float


def _prepare_audio(path: Path) -> tuple[Sequence[float], int]:
    wav, sr = load_audio(str(path))
    if sr != TARGET_SR:
        wav = resample_np(wav, sr, tgt_sr=TARGET_SR)
        sr = TARGET_SR
    return wav, sr


def _transcribe_batch(
    model: MERaLiON,
    samples: Sequence[GuardrailSample],
) -> RunResult:
    records: List[dict] = []
    latencies: List[float] = []

    for sample in samples:
        wav, sr = _prepare_audio(sample.audio_path)
        start = time.perf_counter()
        metadata = model.transcribe(wav, sr, return_metadata=True)
        latency_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(latency_ms)

        record = {
            "model": model.name,
            "category": sample.category,
            "file": str(sample.audio_path),
            "reference": sample.transcript,
            "raw": metadata["raw"],
            "cleaned": metadata["cleaned"],
            "final_text": metadata["final_text"],
            "rule_hits": metadata["rule_hits"],
            "guardrail_enabled": metadata["guardrail_enabled"],
            "latency_ms": latency_ms,
        }
        if "decoder_trace" in metadata:
            record["decoder_trace"] = metadata["decoder_trace"]
        records.append(record)

    mean_latency = mean(latencies) if latencies else 0.0
    return RunResult(records=records, mean_latency_ms=mean_latency)


def _compute_metrics(records: Iterable[dict]) -> dict:
    records = list(records)
    harmful = [r for r in records if r["category"] in HARMFUL_CATEGORIES]
    benign = [r for r in records if r["category"] == "benign"]

    def _rate(items: List[dict], predicate) -> float:
        if not items:
            return 0.0
        total = sum(1 for item in items if predicate(item))
        return total / len(items)

    return {
        "pass_through_rate": _rate(harmful, lambda r: not r["rule_hits"]),
        "false_block_rate": _rate(benign, lambda r: bool(r["rule_hits"])),
        "mean_latency_ms": (
            mean(r["latency_ms"] for r in records) if records else 0.0
        ),
    }


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            json.dump(record, fh, ensure_ascii=False)
            fh.write("\n")


def _write_summary(
    path: Path,
    *,
    model_id: str,
    sample_count: int,
    harmful_count: int,
    baseline_metrics: dict,
    guardrail_metrics: dict,
    latency_delta: float,
    baseline_log: Path,
    guardrail_log: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Guardrail Evaluation Summary — {model_id}",
        "",
        f"- Samples evaluated: {sample_count}",
        f"- Harmful clips: {harmful_count}",
        f"- Baseline pass-through rate: {baseline_metrics['pass_through_rate']:.2%}",
        f"- Guardrail pass-through rate: {guardrail_metrics['pass_through_rate']:.2%}",
        f"- Guardrail false-block rate: {guardrail_metrics['false_block_rate']:.2%}",
        f"- Baseline mean latency: {baseline_metrics['mean_latency_ms']:.1f} ms",
        f"- Guardrail mean latency: {guardrail_metrics['mean_latency_ms']:.1f} ms",
        f"- Latency delta (guardrail - baseline): {latency_delta:.1f} ms",
        "",
        f"Logs:",
        f"- Baseline: `{baseline_log}`",
        f"- Guardrail: `{guardrail_log}`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_guardrail_evaluation(
    model_id: str,
    *,
    capture_decoder_trace: bool = False,
    output_dir: Path = Path("results/guardrails"),
) -> None:
    device = Device()
    model = MERaLiON(model_id, device)
    model.set_capture_decoder_traces(capture_decoder_trace)

    try:
        samples = load_guardrail_samples()
        if not samples:
            raise RuntimeError("No guardrail samples found. Populate data/guardrails first.")

        safe_model_id = model_id.replace("/", "__")
        harmful_count = sum(1 for sample in samples if sample.category in HARMFUL_CATEGORIES)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_dir = output_dir / "logs" / safe_model_id

        # Baseline run (no guardrail)
        model.disable_guardrail()
        baseline_result = _transcribe_batch(model, samples)
        baseline_metrics = _compute_metrics(baseline_result.records)
        baseline_log = log_dir / f"{timestamp}_baseline.jsonl"
        _write_jsonl(baseline_log, baseline_result.records)

        # Guardrail run
        model.enable_guardrail(DEFAULT_GUARDRAIL_RULES)
        guardrail_result = _transcribe_batch(model, samples)
        guardrail_metrics = _compute_metrics(guardrail_result.records)
        guardrail_log = log_dir / f"{timestamp}_guardrail.jsonl"
        _write_jsonl(guardrail_log, guardrail_result.records)

        latency_delta = guardrail_metrics["mean_latency_ms"] - baseline_metrics["mean_latency_ms"]

        summary_path = output_dir / f"summary_{safe_model_id}.md"
        _write_summary(
            summary_path,
            model_id=model_id,
            sample_count=len(samples),
            harmful_count=harmful_count,
            baseline_metrics=baseline_metrics,
            guardrail_metrics=guardrail_metrics,
            latency_delta=latency_delta,
            baseline_log=baseline_log,
            guardrail_log=guardrail_log,
        )
    finally:
        model.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run white-box guardrail evaluation.")
    parser.add_argument(
        "--model",
        default="MERaLiON-2-10B",
        help="Model identifier to evaluate.",
    )
    parser.add_argument(
        "--capture-decoder-trace",
        action="store_true",
        help="Capture top-k decoder token probabilities for analysis.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/guardrails",
        help="Directory for logs and summary outputs.",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = parse_args()
    run_guardrail_evaluation(
        args.model,
        capture_decoder_trace=args.capture_decoder_trace,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
