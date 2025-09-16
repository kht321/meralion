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
`Whisper-small_results.jsonl`. Whisper currently achieves ~0.76 F1 on the toy
set; MERaLiON is still TODO (its processor rejects the audio at present).

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

| Model           | Precision | Recall | F1   | Notes |
|-----------------|-----------|--------|------|-------|
| Whisper-small   | ~1.00     | ~0.62  | ~0.76 | 8/13 profane utterances detected; no hallucinated profanity. |
| MERaLiON-2-10B  | 0.00      | 0.00   | 0.00 | Processor still returns `NoneType.ndim`; no transcripts produced. |

Interpretation: Whisper is conservative—great at avoiding false positives but
still misses five injected terms. Improving recall (prompting, augmentation,
fine-tuning) is the next obvious knob. MERaLiON remains TODO until the
processor path handles the resampled audio correctly.

## Notes & TODOs

- The dataset is intentionally tiny; scale `make_synthetic_profanity.py` for
  broader coverage or point `mvp.py` at a real evaluation set.
- Fix MERaLiON’s preprocessing so it consumes the 16 kHz waveforms; only then
  will its metrics mean anything.
- Keep `soundfile`/`torchaudio` installed—without them the loaders return
  `None` arrays and the processors fail immediately.
