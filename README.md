# MERaLiON Profanity Evaluation MVP

This repository contains a minimal pipeline for stress-testing speech-to-text
models on profanity-heavy audio. The workflow is intentionally small:

1. **Generate synthetic clips** – `make_synthetic_profanity.py` injects terms
   from `profanity_list.txt` into a handful of seed sentences and uses Coqui
   TTS to synthesise audio (written to `data/synthetic_profanity/`).
2. **Benchmark models** – `mvp.py` loads the synthetic dataset, resamples each
   clip to 16 kHz, and evaluates MERaLiON-2-10B and Whisper-small for profanity
   detection precision/recall. Results are written to `results/`.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
pip install -r requirements.txt

# Synthesise the toy dataset (15 clips by default)
python make_synthetic_profanity.py

# Run the benchmark
python mvp.py
```

You should see result files under `results/`, including
`*_results.jsonl`. On the 15-sample toy set both Whisper-small and
MERaLiON-2-10B land around 0.76 F1 after the recent fixes.

## Code walkthrough

### `make_synthetic_profanity.py`
- Loads seed sentences (or `data/NSC/transcripts.txt` if available).
- Pulls replacements from `profanity_list.txt` and injects them with an 80 % probability.
- Synthesises each variant with Coqui TTS (`tts_models/en/ljspeech/fast_pitch`).
- Writes `.wav` clips plus a `metadata.jsonl` manifest under `data/synthetic_profanity/`.

### `mvp.py`
- Loads `data/synthetic_profanity/metadata.jsonl` via `datasets.load_dataset`.
- Reads every `.wav`, resamples to **16 kHz** (Whisper/MERaLiON requirement) using
  torchaudio when available, otherwise NumPy interpolation.
- Runs the appropriate `AutoProcessor` and `model.generate`, then flags profanity
  in references/predictions with the shared regex.
- Persists row-by-row logs (`*_results.jsonl` / `.csv`) and aggregated metrics
  (`results/summary.json`).

## Repo structure

```
├── make_synthetic_profanity.py   # synthetic audio generator
├── mvp.py                        # evaluation script
├── profanity_list.txt            # lexicon injected into the audio
├── data/                         # synthetic outputs live here
└── results/                      # JSON/CSV logs after benchmarking
```

## Latest results (15-clip toy set)

| Model             | Precision | Recall | F1    | Notes |
|-------------------|-----------|--------|-------|-------|
| Whisper-small     | 1.00      | 0.62   | 0.76  | Flags 8 of 13 profane utterances; never hallucinates profanity. |
| MERaLiON-2-10B    | 1.00      | 0.62   | 0.76  | Matches Whisper on recall; outputs still prepend the `model …` tag. |

Interpretation: both models are very conservative—no false positives, but they
still miss five injected terms. Better prompting or fine-tuning for profanity
recall remains the obvious next knob, and MERaLiON could use a light
post-processing step to strip its chat prefix.

## Notes & TODOs

- The dataset is intentionally tiny; scale `make_synthetic_profanity.py` for
  broader coverage or point `mvp.py` at a real evaluation set.
- Strip MERaLiON’s residual `model <speaker…>` prefix before scoring, or switch
  to a prompt that avoids the role tag entirely.
- Keep `soundfile`/`torchaudio` installed—without them the loaders return
  `None` arrays and the processors fail immediately.

## Evaluation Roadmap (in progress)

- **Core fairness & robustness audit**
  - Slice NSC (with demographics) + Singlish test sets to report WER/CER, profanity miss rate, and confidence calibration for MERaLiON vs Whisper.
  - Run Speech Robust Bench perturbations (noise, reverberation, codec, tempo) to quantify relative degradation.
  - Track Singlish-specific markers (particles like `lah`, local proper nouns) to surface omissions or standard-English substitutions.

- **Safety probing**
  - Probe guardrail gaps with targeted profanity / slur triggers and note hallucinations or misses.
  - Exercise universal-ish perturbations (band-limited noise, time stretch) to gauge how easily profanities slip through.
  - Catalogue failure modes for safety-critical domains (medical, legal) using domain-specific vocab lists.

- **Extension ideas**
  1. Fine-tune a lightweight guardrail (e.g., post-decoder classifier) and report deltas.
  2. Curate additional conversational corpora (podcasts, rallies, vlogs) for qualitative fairness audits once core metrics are stable.
  3. Add demographic-aware confidence calibration analysis to flag over- and under-confident subgroups.
