"""Script to compute WER/CER broken down by demographic groups (fairness).

Usage:
    python -m asr_eval.eval.run_fairness --config configs/fairness.yaml

The config is a YAML with keys similar to robustness config:
- dataset_manifest: path to JSONL manifest (or CSV) with audio paths and text
- dataset_audio_dir: optional root for audio files
- results_dir: where to write results (defaults to results/fairness)
- group_by: list of metadata columns to group by (e.g., [race, gender, age])
- models: list of model names
- bootstrap: {n_samples: 1000, alpha: 0.05}
"""

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
from ..metrics.ci import bootstrap_mean_ci
from ..metrics.wer_cer import corpus_scores, Scores
from ..models.meralion import MERaLiON
from ..models.whisper import Whisper
from ..device import Device


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
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # fallback: try CSV via pandas
                break

    if not rows:
        # Try CSV table with audio,text and metadata columns
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            rows.append(r.to_dict())

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


def _prepare_audio(wav: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    if sr != TARGET_SR:
        wav = resample_np(wav, sr, TARGET_SR)
        sr = TARGET_SR
    return wav, sr


def _transcribe_full_audio(model, wav: np.ndarray, sr: int, max_chunk_seconds: float | None, overlap_seconds: float) -> str:
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


def run(cfg_path: Path) -> None:
    cfg = yaml.safe_load(cfg_path.read_text())

    results_dir = Path(cfg.get("results_dir", "results/fairness"))
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(cfg["dataset_manifest"])
    audio_root = Path(cfg.get("dataset_audio_dir", "."))
    text_field = cfg.get("text_field", "text")
    group_by = cfg.get("group_by", ["speaker"]) or ["speaker"]

    manifest = load_manifest(manifest_path)

    device = Device()

    max_chunk_seconds = cfg.get("max_chunk_seconds")
    if max_chunk_seconds is not None:
        max_chunk_seconds = float(max_chunk_seconds)
        if max_chunk_seconds <= 0:
            max_chunk_seconds = None
    chunk_overlap_seconds = float(cfg.get("chunk_overlap_seconds", 2.0))

    all_summary_rows: List[Dict[str, Any]] = []

    for model_name in cfg.get("models", []):
        model = build_model(model_name, device)
        try:
            for seed in cfg.get("seeds", [0]):
                seed = int(seed)
                set_all_seeds(seed)

                refs: List[str] = []
                waves: List[Tuple[np.ndarray, int]] = []
                paths: List[str] = []
                rows_meta: List[Dict[str, Any]] = []

                for row in manifest:
                    rel = _extract_audio_path(row)
                    if not rel:
                        continue
                    path = audio_root / rel
                    try:
                        wav, sr = load_audio(str(path))
                    except Exception:
                        # Skip if audio can't be loaded
                        continue
                    wav, sr = _prepare_audio(wav, sr)
                    refs.append((row.get(text_field) or "").strip())
                    waves.append((wav, sr))
                    paths.append(str(path))
                    rows_meta.append(row)

                hyps: List[str] = []
                for idx, (wav, sr) in enumerate(
                    tqdm(waves, desc=f"{model_name}/seed{seed}", leave=False)
                ):
                    hyp = _transcribe_full_audio(
                        model,
                        wav,
                        sr,
                        max_chunk_seconds,
                        chunk_overlap_seconds,
                    )
                    hyps.append(hyp)

                from jiwer import wer, cer

                per_utt_wers: List[float] = []
                per_utt_cers: List[float] = []
                for ref, hyp in zip(refs, hyps):
                    per_utt_wers.append(wer(ref or "", hyp or ""))
                    per_utt_cers.append(cer(ref or "", hyp or ""))

                corpus = corpus_scores(refs, hyps) if refs and hyps else Scores(wer=float('nan'), cer=float('nan'), n_utts=0)

                # combine metadata with per-utt metrics
                rows: List[Dict[str, Any]] = []
                for meta, path, ref, hyp, u_wer, u_cer in zip(rows_meta, paths, refs, hyps, per_utt_wers, per_utt_cers):
                    row_out = dict(meta)
                    row_out.update(
                        {
                            "model": model_name,
                            "seed": seed,
                            "path": path,
                            "ref_text": ref,
                            "pred_text": hyp,
                            "wer": float(u_wer),
                            "cer": float(u_cer),
                        }
                    )
                    rows.append(row_out)

                df = pd.DataFrame(rows)

                # write per-utterance results
                per_utt_path = results_dir / f"{model_name}_seed{seed}_per_utt.csv"
                df.to_csv(per_utt_path, index=False)

                # aggregate by group_by columns
                agg_rows: List[Dict[str, Any]] = []
                bootstrap_cfg = cfg.get("bootstrap", {})
                n_samples = int(bootstrap_cfg.get("n_samples", 1000))
                alpha = float(bootstrap_cfg.get("alpha", 0.05))

                if not group_by:
                    group_by = ["speaker"]

                for group_keys, group_df in df.groupby(group_by):
                    # group_keys may be a scalar or tuple
                    if not isinstance(group_keys, tuple):
                        group_keys = (group_keys,)

                    g_wer_mean, g_wer_lo, g_wer_hi = bootstrap_mean_ci(
                        group_df["wer"].to_numpy(), n_samples, alpha, seed=0
                    )
                    g_cer_mean, g_cer_lo, g_cer_hi = bootstrap_mean_ci(
                        group_df["cer"].to_numpy(), n_samples, alpha, seed=1
                    )

                    entry: Dict[str, Any] = {
                        "model": model_name,
                        "seed": seed,
                        "n_utts": int(group_df.shape[0]),
                        "wer_mean": float(g_wer_mean),
                        "wer_lo": float(g_wer_lo),
                        "wer_hi": float(g_wer_hi),
                        "cer_mean": float(g_cer_mean),
                        "cer_lo": float(g_cer_lo),
                        "cer_hi": float(g_cer_hi),
                    }

                    # attach group key columns
                    for col, val in zip(group_by, group_keys):
                        entry[col] = val

                    agg_rows.append(entry)

                agg_df = pd.DataFrame(agg_rows)
                agg_path = results_dir / f"{model_name}_seed{seed}_grouped.csv"
                agg_df.to_csv(agg_path, index=False)

                all_summary_rows.extend(rows)
        finally:
            model.close()

    # write global per-utt results for all models/seeds
    global_per_utt = pd.DataFrame(all_summary_rows)
    if not global_per_utt.empty:
        global_per_utt_path = results_dir / "all_per_utt.csv"
        global_per_utt.to_csv(global_per_utt_path, index=False)

    print(f"Wrote fairness results to {results_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    run(args.config)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
