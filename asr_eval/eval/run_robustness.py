"""Script to benchmark ASR robustness under audio corruptions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from ..audio import TARGET_SR, load_audio, resample_np
from ..corruption.grid import expand_grid
from ..corruption.ops import REGISTRY as CORR_REGISTRY
from ..device import Device
from ..metrics.ci import bootstrap_mean_ci
from ..metrics.wer_cer import corpus_scores
from ..models.meralion import MERaLiON
from ..models.whisper import Whisper

ALIASES = {
    "meralion-2-10b": "MERaLiON/MERaLiON-2-10B",
    "meralion-2-3b": "MERaLiON/MERaLiON-2-3B",
    "whisper-small": "openai/whisper-small",
}


def build_model(name: str, device: Device):
    canonical = name.lower().strip()
    if canonical in {"meralion", "meralion-2-10b"}:
        return MERaLiON(ALIASES["meralion-2-10b"], device)
    if canonical == "meralion-2-3b":
        return MERaLiON(ALIASES["meralion-2-3b"], device)
    if canonical in {"whisper", "whisper-small"}:
        return Whisper(ALIASES["whisper-small"], device)
    if "meralion" in canonical:
        return MERaLiON(name, device)
    return Whisper(name, device)


def set_all_seeds(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_audio_path(row: Dict[str, Any]) -> str | None:
    audio = row.get("audio")
    if isinstance(audio, dict):
        return audio.get("path") or audio.get("audio_path")
    if isinstance(audio, str):
        return audio
    fallback = row.get("audio_path")
    if isinstance(fallback, str):
        return fallback
    return None


def _corruption_kwargs(name: str, severity: Any) -> Dict[str, Any]:
    match name:
        case "noise_snr_db":
            return {"snr_db": float(severity)}
        case "speed":
            return {"factor": float(severity)}
        case "pitch_semitones":
            return {"semitones": float(severity)}
        case "reverb_decay":
            return {"decay": float(severity)}
        case "clipping_ratio":
            return {"ratio": float(severity)}
        case _:
            return {}


def _prepare_audio(wav: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    if sr != TARGET_SR:
        wav = resample_np(wav, sr, TARGET_SR)
        sr = TARGET_SR
    return wav, sr


def _transcribe_full_audio(
    model,
    wav: np.ndarray,
    sr: int,
    max_chunk_seconds: float | None,
    overlap_seconds: float,
) -> str:
    if not max_chunk_seconds or max_chunk_seconds <= 0:
        return model.transcribe(wav, sr)

    duration = wav.shape[0] / float(sr)
    if duration <= max_chunk_seconds:
        return model.transcribe(wav, sr)

    max_samples = max(int(max_chunk_seconds * sr), 1)
    overlap_samples = max(int(overlap_seconds * sr), 0)
    if overlap_samples >= max_samples:
        overlap_samples = max_samples // 4
    step = max(max_samples - overlap_samples, 1)

    transcripts: List[str] = []
    start = 0
    total = wav.shape[0]
    while start < total:
        end = min(start + max_samples, total)
        chunk = wav[start:end]
        transcripts.append(model.transcribe(chunk, sr))
        if end >= total:
            break
        start += step

    return " ".join(t.strip() for t in transcripts if t.strip())


def _trim_transcript(hyp: str, ref: str, do_trim: bool) -> str:
    if not do_trim:
        return hyp

    ref = ref.strip()
    if not ref:
        return hyp

    ref_tokens = ref.split()
    if not ref_tokens:
        return hyp

    hyp_tokens = hyp.strip().split()
    if len(hyp_tokens) <= len(ref_tokens):
        return hyp.strip()

    trimmed = " ".join(hyp_tokens[: len(ref_tokens)])
    return trimmed


def run(cfg_path: Path, emit_jsonl: bool = False) -> None:
    cfg = yaml.safe_load(cfg_path.read_text())

    results_dir = Path(cfg.get("results_dir", "results/robustness"))
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(cfg["dataset_manifest"])
    audio_root = Path(cfg.get("dataset_audio_dir", "."))
    text_field = cfg.get("text_field", "text")

    manifest = load_manifest(manifest_path)
    grid = expand_grid(cfg.get("corruptions", {}))

    device = Device()

    details_path = results_dir / "details.jsonl"
    details_f = details_path.open("w") if emit_jsonl else None

    summary_rows: List[Dict[str, Any]] = []

    max_chunk_seconds = cfg.get("max_chunk_seconds")
    if max_chunk_seconds is not None:
        max_chunk_seconds = float(max_chunk_seconds)
        if max_chunk_seconds <= 0:
            max_chunk_seconds = None
    chunk_overlap_seconds = float(cfg.get("chunk_overlap_seconds", 2.0))
    trim_to_ref_tokens = bool(cfg.get("trim_to_ref_tokens", False))

    for model_name in cfg.get("models", []):
        model = build_model(model_name, device)
        try:
            for seed in cfg.get("seeds", [0]):
                seed = int(seed)
                set_all_seeds(seed)

                refs: List[str] = []
                waves: List[Tuple[np.ndarray, int]] = []
                paths: List[str] = []

                for row in manifest:
                    rel = _extract_audio_path(row)
                    if not rel:
                        continue
                    path = audio_root / rel
                    wav, sr = load_audio(str(path))
                    wav, sr = _prepare_audio(wav, sr)
                    refs.append((row.get(text_field) or "").strip())
                    waves.append((wav, sr))
                    paths.append(str(path))

                # baseline
                hyps_clean: List[str] = []
                for idx, (wav, sr) in enumerate(
                    tqdm(
                        waves,
                        desc=f"{model_name}/seed{seed}/clean",
                        leave=False,
                    )
                ):
                    hyp = _transcribe_full_audio(
                        model,
                        wav,
                        sr,
                        max_chunk_seconds,
                        chunk_overlap_seconds,
                    )
                    hyps_clean.append(
                        _trim_transcript(hyp, refs[idx], trim_to_ref_tokens)
                    )

                clean_scores = corpus_scores(refs, hyps_clean)

                for corr_name, meta in grid:
                    func = CORR_REGISTRY[corr_name]
                    severity = meta.get("severity")
                    kwargs = _corruption_kwargs(corr_name, severity)
                    corrupted_hyps: List[str] = []

                    for idx, (wav, sr) in enumerate(
                        tqdm(
                            waves,
                            desc=f"{model_name}/seed{seed}/{corr_name}:{severity}",
                            leave=False,
                        )
                    ):
                        wav2 = func(
                            wav,
                            sr,
                            seed=seed,
                            **kwargs,
                        )
                        hyp = _transcribe_full_audio(
                            model,
                            wav2,
                            sr,
                            max_chunk_seconds,
                            chunk_overlap_seconds,
                        )
                        corrupted_hyps.append(
                            _trim_transcript(hyp, refs[idx], trim_to_ref_tokens)
                        )

                    scores = corpus_scores(refs, corrupted_hyps)

                    if details_f is not None:
                        for path, ref, hyp in zip(paths, refs, corrupted_hyps):
                            details_f.write(
                                json.dumps(
                                    {
                                        "model": model_name,
                                        "seed": seed,
                                        "corruption": corr_name,
                                        "severity": severity,
                                        "path": path,
                                        "ref": ref,
                                        "hyp": hyp,
                                    }
                                )
                                + "\n"
                            )

                    summary_rows.append(
                        {
                            "model": model_name,
                            "seed": seed,
                            "corruption": corr_name,
                            "severity": severity,
                            "wer": scores.wer,
                            "cer": scores.cer,
                            "n_utts": scores.n_utts,
                            "wer_delta_vs_clean": scores.wer - clean_scores.wer,
                            "cer_delta_vs_clean": scores.cer - clean_scores.cer,
                        }
                    )

                    # Save checkpoint after each corruption completes
                    checkpoint = pd.DataFrame(summary_rows)
                    checkpoint_path = results_dir / "per_seed_checkpoint.csv"
                    checkpoint.to_csv(checkpoint_path, index=False)
        finally:
            model.close()

    if details_f is not None:
        details_f.close()

    per_seed = pd.DataFrame(summary_rows)
    raw_path = results_dir / "per_seed.csv"
    per_seed.to_csv(raw_path, index=False)

    agg_rows: List[Dict[str, Any]] = []
    bootstrap_cfg = cfg.get("bootstrap", {})
    n_samples = int(bootstrap_cfg.get("n_samples", 1000))
    alpha = float(bootstrap_cfg.get("alpha", 0.05))

    for (model_name, corr_name, severity), group in per_seed.groupby(
        ["model", "corruption", "severity"]
    ):
        wer_mean, wer_lo, wer_hi = bootstrap_mean_ci(
            group["wer"].to_numpy(), n_samples, alpha, seed=0
        )
        cer_mean, cer_lo, cer_hi = bootstrap_mean_ci(
            group["cer"].to_numpy(), n_samples, alpha, seed=1
        )
        dwer_mean, dwer_lo, dwer_hi = bootstrap_mean_ci(
            group["wer_delta_vs_clean"].to_numpy(), n_samples, alpha, seed=2
        )
        dcer_mean, dcer_lo, dcer_hi = bootstrap_mean_ci(
            group["cer_delta_vs_clean"].to_numpy(), n_samples, alpha, seed=3
        )

        agg_rows.append(
            {
                "model": model_name,
                "corruption": corr_name,
                "severity": severity,
                "wer_mean": wer_mean,
                "wer_lo": wer_lo,
                "wer_hi": wer_hi,
                "cer_mean": cer_mean,
                "cer_lo": cer_lo,
                "cer_hi": cer_hi,
                "dWER_mean": dwer_mean,
                "dWER_lo": dwer_lo,
                "dWER_hi": dwer_hi,
                "dCER_mean": dcer_mean,
                "dCER_lo": dcer_lo,
                "dCER_hi": dcer_hi,
                "n_seed": int(group["seed"].nunique()),
                "n_utts": int(group["n_utts"].max()),
            }
        )

    agg = pd.DataFrame(agg_rows)
    summary_path = results_dir / "summary.csv"
    agg.to_csv(summary_path, index=False)

    print(f"wrote {raw_path} and {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--emit_jsonl", action="store_true")
    args = parser.parse_args()

    run(args.config, emit_jsonl=args.emit_jsonl)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
