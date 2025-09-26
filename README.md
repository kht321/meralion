# MERaLiON ASR Safety Evaluation Toolkit

This repository implements the code behind our proposal to assess the safety of
MERaLiON’s automatic speech recognition (ASR) models—focusing on robustness,
fairness, and (later) backdoor resilience. The goal is to give both technical
and non-technical readers a clear view of what data is used, how metrics are
computed, and how to reproduce the experiments.

---

## Why this project exists

ASR systems convert audio into text. When they fail, the errors can propagate to
legal transcripts, medical notes, or moderation pipelines and cause harm. The
MERaLiON family of models is optimised for Singaporean and Southeast Asian
speech, but safety considerations were explicitly deferred to downstream users.
Our evaluation focuses on three questions:

1. **Robustness:** Do transcripts stay accurate when we add realistic acoustic
   distortions such as MRT background noise or reverberation from HDB flats?
2. **Fairness:** Are errors evenly distributed across speakers with different
   demographics, accents, or recording devices?
3. **Backdoor resilience (stretch goal):** Can small, malicious fine-tuning
   trigger targeted mistranscriptions without degrading normal accuracy?

This repository currently delivers the robustness pipeline and the supporting
data tooling; fairness analysis scaffolding is described below and will be
implemented next, followed by backdoor experiments.

---

## Repository tour

```
├── asr_eval/                    # Python package with all evaluation logic
│   ├── audio.py                 # Audio loading + resampling utilities
│   ├── cli.py                   # Command-line entry point (asr-eval)
│   ├── corruption/              # Deterministic audio perturbations
│   ├── datasets/                # NSC manifest builders
│   ├── eval/                    # Experiment drivers (robustness, …)
│   ├── metrics/                 # WER/CER scoring + bootstrap CI helpers
│   └── models/                  # Thin wrappers for MERaLiON & Whisper
├── configs/                     # YAML configs that define experiment grids
├── data/                        # Local datasets (ignored in git)
│   └── manifests/               # Generated JSONL manifests (tracked)
├── results/                     # Output tables / plots (local)
├── Makefile                     # Convenience targets (manifest, test, robust)
└── README.md                    # You are here
```

**Historical note:** the earlier `mvp.py` prototype is **no longer used** and is
not part of the documented workflow.

---

## Datasets

### National Speech Corpus (NSC)

The NSC is Singapore’s flagship speech dataset (≈10 000 hours, >1 000 speakers)
released by IMDA. It is organised into parts that capture different speaking
contexts—phonetically balanced scripts, conversations, code-switching with
Mandarin/Malay/Tamil, expressive speech, debates, and scenario-based recordings.
Each part ships with speaker metadata (gender, ethnicity, age band, education,
device) in Excel spreadsheets.

- Place the raw NSC files under `data/NSC/` following IMDA’s directory
  structure (not tracked in git).
- Speaker metadata spreadsheets live in `data/NSC/Metadata/`.
- Run `make nsc-manifest` to generate `data/manifests/nsc_part1.jsonl`, a JSONL
  manifest covering Part 1 (local accents) with columns:
  - `audio.path`: relative WAV path
  - `text`: transcript from the official `.TXT` files (normalised lowercase)
  - `speaker`, `session`: speaker/session identifiers from the directory tree
  - `part`: flag identifying the corpus slice (e.g., `part1_local_accents`)

Future manifests will extend this to other NSC parts and join speaker
metadata (gender, ethnicity, device, etc.) so fairness metrics can be computed
per subgroup.

### Synthetic profanity toy set (optional)

A tiny TTS-generated dataset (`data/synthetic_profanity/`) is included for quick
smoke tests. It is **not** used in the main evaluation pipeline but remains
useful for validating installation without NSC access.

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e . pytest
```

Optional: install any additional tools listed in `requirements.txt` if you plan
to recreate the TTS dataset or run extended analyses.

---

## Workflow overview

### 1. Generate manifests

```bash
make nsc-manifest
```

This parses NSC Part 1 transcript files, pairs them with WAV paths, and writes a
JSONL manifest at `data/manifests/nsc_part1.jsonl`. The manifest is lightweight
and checked into the repository so collaborators without local NSC audio can see
its schema (the audio itself stays local).

### 2. Run unit checks

```bash
make test
```

Currently this exercises deterministic corruption noise and the manifest builder
(smoke test for at least one NSC utterance). Expand with additional tests as the
project grows.

### 3. Robustness sweep

```bash
make robust
```

The command executes `python -m asr_eval.eval.run_robustness` with the default
configuration in `configs/robustness.yaml`. The pipeline:

1. Loads the NSC manifest and pre-fetches the clean waveforms to avoid repeated
   disk I/O.
2. Builds the ASR model selected in the config (MERaLiON-2-10B and
   Whisper-small by default) using the device auto-selector (`asr_eval/device.py`).
3. Computes baseline transcripts and WER/CER scores on clean audio.
4. Applies each corruption from the grid (noise SNR, speed change, pitch shift,
   reverberation decay, clipping) at multiple severity levels. All perturbations
   are deterministic given the seed so experiments are reproducible.
5. Records per-seed, per-corruption metrics (`results/robustness/per_seed.csv`)
   and bootstrap aggregates with 95 % confidence intervals (`summary.csv`).
6. Optionally emits JSONL logs of every utterance-level transcript when
   `--emit_jsonl` is passed (enabled in the Makefile target).

#### Adjusting the robustness config

`configs/robustness.yaml` controls the experiment:

- `dataset_manifest`, `dataset_audio_dir`: point to the desired split (clone the
  file for other NSC parts or external datasets).
- `models`: list of aliases or Hugging Face IDs (mapped through `ALIASES`).
- `seeds`: seeds for Monte Carlo runs to capture variation.
- `corruptions`: corruption grid. Extend or replace with local recordings (e.g.,
  MRT cabin impulse responses) to address realism risks.
- `bootstrap`: number of samples and confidence level for the aggregated table.

### 4. Fairness analysis (coming soon)

The fairness plan builds on the same manifests by joining metadata from the
Excel spreadsheets. Steps (to be implemented next):

1. Convert the metadata spreadsheets into structured CSV/Parquet files and join
   them onto the manifest rows by speaker ID.
2. Stratify evaluation sets across gender, ethnicity, age, and recording device
   while balancing SNR/domain to avoid confounding (per the risk log).
3. Reuse `asr_eval.metrics.wer_cer` to compute WER/CER per subgroup and report
   max–min gaps, medians, and tail percentiles with statistical tests (e.g.,
   Welch’s t-test, ANOVA) and multiple-comparison control (Benjamini–Hochberg).
4. Extend reporting to include calibration curves (ECE/Brier) once model
   confidences are available.

Placeholder notebooks and scripts will be added under `analysis/`.

### 5. Backdoor evaluation (stretch goal)

Later milestones will explore parameter-efficient fine-tuning (LoRA/PEFT) with
poisoned audio triggers. The repository already contains device-aware model
wrappers and corruption tooling that can be repurposed for trigger insertion and
attack success measurement (BD-ASR%). Implementation details will follow once
robustness and fairness foundations are complete.

---

## Command cheat sheet

| Purpose                    | Command                                                                           |
|--------------------------- |-----------------------------------------------------------------------------------|
| Create NSC manifest        | `make nsc-manifest`                                                                |
| Run tests                  | `make test`                                                                       |
| Robustness evaluation      | `make robust`                                                                     |
| Custom robustness run      | `python -m asr_eval.eval.run_robustness --config path/to/config.yaml [--emit_jsonl]` |
| Generate manifest manually | `python -m asr_eval.datasets.nsc_manifest --nsc-root data/NSC --output data/manifests/nsc_part1.jsonl` |

---

## Outputs

- `results/robustness/per_seed.csv`: WER/CER for every (model, seed, corruption,
  severity) tuple.
- `results/robustness/summary.csv`: bootstrap-aggregated means and 95 % CIs,
  including deltas relative to clean audio.
- `results/robustness/details.jsonl`: (optional) row-by-row transcripts for
  deeper audits.

For now, the “Results” section in reports/papers can link to these CSVs; summary
plots will be generated once fairness analysis and backdoor experiments are
integrated.

---

## Risk mitigation checklist

The proposal identified several methodological risks (corruption realism,
train–eval contamination, demographic confounds, etc.). The repository addresses
some immediately:

- **Reproducibility:** All corruptions use deterministic seeds; clean baselines
  are cached; configs store every knob.
- **Dataset governance:** Heavy NSC assets stay local (`data/NSC/` is
  `.gitignore`’d). Only derived manifests are tracked.
- **Future-proofing:** Hooks are in place for metadata joins and for plugging in
  additional corruption sources (local impulse responses, device codecs).

Outstanding items (interaction effects, multiple comparisons, calibration) are
tracked in the project roadmap and will be implemented alongside fairness and
backdoor work.

---

## Contributing / extending

1. Open an issue or draft proposal for new corruptions, fairness features, or
   backdoor experiments.
2. Add or update manifests/configs under `data/manifests/` and `configs/`.
3. Ensure `make test` passes and document new user-facing behaviour in this
   README.
4. When adding results, place artefacts in `results/` and describe the findings
   in a dedicated section (reserved for future updates).

---

## Results (placeholder)

To be updated once the first full robustness and fairness sweeps are complete.

---

## References

- Ardila et al. (2019). *Common Voice: a Massively-Multilingual Speech Corpus.*
- MERaLiON team (2025a–c). *Model release, AudioBench leaderboard, and docs.*
- Panayotov et al. (2015). *LibriSpeech: An ASR corpus based on public domain
  audio books.*
- Rauh et al. (2024). *Gaps in the Safety Evaluation of Generative AI.*
- Shah et al. (2024). *Speech Robust Bench: A robustness benchmark for speech
  recognition.*

