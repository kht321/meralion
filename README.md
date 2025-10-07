# MERaLiON ASR Safety Evaluation Toolkit

This repository implements the code behind our proposal to assess the safety of
MERaLiON's automatic speech recognition (ASR) models—focusing on robustness
and fairness. The goal is to give both technical and non-technical readers a
clear view of what data is used, how metrics are computed, and how to reproduce
the experiments.

---

## Why this project exists

ASR systems convert audio into text. When they fail, the errors can propagate to
legal transcripts, medical notes, or moderation pipelines and cause harm. The
MERaLiON family of models is optimised for Singaporean and Southeast Asian
speech, but safety considerations were explicitly deferred to downstream users.
Our evaluation focuses on two questions:

1. **Robustness:** Do transcripts stay accurate when we add realistic acoustic
   distortions such as MRT background noise or reverberation from HDB flats?
2. **Fairness:** Are errors evenly distributed across speakers with different
   demographics, accents, or recording devices?

This repository currently delivers the robustness pipeline and the supporting
data tooling; fairness analysis scaffolding is described below and will be
implemented next.

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
  - Each manifest entry corresponds to a short clip (2.1 s – 9.4 s, mean 4.8 s);
    the default Part 1 manifest contains 682 utterances.

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

> **Workload overview:** The default manifest (`data/manifests/nsc_part1.jsonl`)
> contains 682 utterances. Combined with the default configuration (2 models × 3
> seeds × 6 corruption settings—clean + five perturbations), the job produces
> 24 552 individual transcriptions before aggregating metrics. Adjusting any of
> these knobs scales runtime roughly linearly.

#### Adjusting the robustness config

`configs/robustness.yaml` controls the experiment:

- `dataset_manifest`, `dataset_audio_dir`: point to the desired split (clone the
  file for other NSC parts or external datasets).
- `models`: list of aliases or Hugging Face IDs (mapped through `ALIASES`).
- `seeds`: seeds for Monte Carlo runs to capture variation.
- `corruptions`: corruption grid. Extend or replace with local recordings (e.g.,
  MRT cabin impulse responses) to address realism risks.
- `bootstrap`: number of samples and confidence level for the aggregated table.

---

## Methodology & evaluation strategy

### Robustness (current focus)

- **Rationale:** Real deployments rarely enjoy clean studio audio. We mirror the
  corruption taxonomy from Speech Robust Bench (noise, speed, pitch,
  reverberation, clipping) and tune severities to local soundscapes such as MRT
  cabins or HDB living rooms. Deterministic seeds let us measure *relative*
  degradation against the cached clean baseline.
- **Evaluation:** For each (model, corruption, severity, seed) tuple we record
  WER, CER, and deltas versus clean audio, then bootstrap across seeds for 95 %
  confidence intervals. Interaction tests (noise×reverb, noise×speed) are on the
  roadmap to address the “ignoring interaction effects” risk.

### Fairness (in flight)

- **Rationale:** Apparent demographic gaps can be confounded by microphone
  choice or ambient noise. By joining NSC speaker metadata and building a
  lockbox test set of public social clips, we can separate demographic effects
  from acoustic ones.
- **Planned evaluation:**
  - Construct balanced manifests for gender, ethnicity, age, and device categories.
  - Report per-group WER/CER medians and tail percentiles (P90/P95) alongside
    max–min gaps.
  - Use Welch’s t-tests or ANOVA with Benjamini–Hochberg FDR control, and extend
    to calibration metrics (ECE/Brier) once token confidences are available.

### 3. Fairness analysis (coming soon)

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

For now, the "Results" section in reports/papers can link to these CSVs; summary
plots will be generated once fairness analysis is integrated.

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
tracked in the project roadmap and will be implemented alongside fairness work.

---

## Contributing / extending

1. Open an issue or draft proposal for new corruptions or fairness features.
2. Add or update manifests/configs under `data/manifests/` and `configs/`.
3. Ensure `make test` passes and document new user-facing behaviour in this
   README.
4. When adding results, place artefacts in `results/` and describe the findings
   in a dedicated section (reserved for future updates).

---

## Results (NSC Part 1 robustness sweep)

Latest run: MERaLiON-2-10B and Whisper-small evaluated on 682 utterances from
NSC Part 1 under clean audio and five corruption types (noise, speed, pitch,
reverb, clipping) across three seeds per setting. Key aggregates from
`results/robustness/summary.csv`:

| Model            | Clean WER | Clean CER | Worst ΔWER | Worst corruption (WER) | Observations |
|------------------|-----------|-----------|------------|------------------------|--------------|
| MERaLiON-2-10B   | 26.1 %    | 22.1 %    | +6.2 pp    | Noise SNR 10 dB (32.3 %) | Mild speed/pitch shifts give a small boost (≈–1 pp); clipping and moderate reverb change <1 pp. |
| MERaLiON-2-3B    | 29.0 %    | 29.3 %    | +5.2 pp    | Noise SNR 10 dB (34.2 %) | Very robust to clipping (0 pp change); noise at 10 dB adds +5.2 pp. All other corruptions stay within +1.1 pp. |
| Whisper-small    | 17.9 %    | 6.1 %     | +77.6 pp   | Reverb decay 0.8 (95.5 %) | Strong reverb severely degrades performance; light noise (30 dB) adds ~0.4 pp, while 10 dB noise adds 18.8 pp. |

### Insights

- **Baseline accuracy:** Whisper-small outperforms both MERaLiON variants on clean
  NSC Part 1 (WER 17.9 % vs. 26.1 % for 10B and 29.0 % for 3B), driven by a
  substantially lower character error rate (6.1 % vs. 22.1 % and 29.3 % respectively).

- **Model size impact (MERaLiON family):** Despite being trained on 40 % of the NSC
  dataset, MERaLiON-2-3B shows only a +2.9 pp WER degradation compared to the 10B
  variant on clean audio. However, the 3B model exhibits significantly higher CER
  (+7.2 pp), suggesting the smaller model struggles more with character-level precision
  while maintaining reasonable word-level accuracy. Notably, the 3B model demonstrates
  slightly better noise robustness (worst ΔWER +5.2 pp vs. +6.2 pp for 10B), indicating
  that data familiarity (NSC training) and robustness scale differently with model size.

- **Noise robustness:** All models handle 30 dB noise well (+0.9 pp WER or less).
  Both MERaLiON variants tolerate 20 dB with minimal degradation (+1.1 pp or less),
  whereas Whisper's WER climbs +2.8 pp. At a challenging 10 dB SNR, MERaLiON-2-10B
  reaches 32.3 % (+6.2 pp), MERaLiON-2-3B reaches 34.2 % (+5.2 pp), while Whisper
  jumps to 36.7 % (+18.8 pp). This demonstrates MERaLiON's superior noise resilience
  despite higher clean error rates, with the 3B variant unexpectedly showing the best
  noise tolerance relative to its clean baseline.

- **Reverberation sensitivity:** Whisper's performance collapses under heavy
  reverberation (WER 95.5 %, CER 75 %), while both MERaLiON models exhibit only
  modest drift (+1.2 pp for 10B, +1.1 pp for 3B at decay 0.8). This suggests
  Whisper's decoder is particularly brittle to long-tail room responses and should
  be a priority for augmentation.

- **Tempo/pitch:** Small speed reductions (0.8×) or pitch shifts (±2 semitones)
  slightly *improve* MERaLiON scores (≈–1 pp for both variants), hinting at
  beneficial regularity in the perturbations. Whisper remains largely unchanged
  across these settings.

- **Amplitude clipping:** Hard clipping shows negligible impact on all models, with
  MERaLiON-2-3B achieving perfect robustness (0 pp change across all clipping ratios).
  This implies that peak amplitudes in NSC Part 1 rarely hit those thresholds.

Full per-seed metrics and bootstrap confidence intervals are stored in
`results/robustness/per_seed.csv` and `results/robustness/summary.csv`.

---

## Results (Self-Curated Conversational Dataset)

Evaluated all 3 models on 2 conversational audio files (test1.mp3, test2.mp3) with manual ground truth transcripts covering the first ~30 seconds of each file. Model transcripts were trimmed to match reference token count to ensure fair comparison. These samples contain multi-speaker Singlish conversations with code-switching, colloquialisms, and disfluencies. Results from `results/self_curated/summary.csv`:

| Model            | Clean WER | Clean CER | Worst ΔWER | Worst corruption (WER) | Best improvement (ΔWER) | Observations |
|------------------|-----------|-----------|------------|------------------------|-------------------------|--------------|
| MERaLiON-2-10B   | 58.0 %    | 29.5 %    | +7.5 pp   | Speed 0.8x (65.5 %)   | -5.2 pp (Pitch +2 semitones) | Severe degradation on conversational speech; slowed playback degrades further while pitch shifts help. |
| MERaLiON-2-3B    | 33.9 %    | 23.9 %    | +21.6 pp  | Noise SNR 10 dB (55.5 %) | -3.3 pp (Reverb 0.8 decay) | Dramatically outperforms 10B by 24.1 pp; extremely vulnerable to noise on conversational data. |
| Whisper-small    | 58.6 %    | 52.3 %    | +24.4 pp  | Reverb decay 0.8 (83.1 %) | -6.5 pp (Speed 1.1x) | Poor baseline on conversational Singlish; reverb catastrophic, faster playback improves performance. |

### Key Observations (Conversational Speech)

- **Domain shift impact:** All models perform dramatically worse on conversational Singlish compared to NSC read speech. MERaLiON-2-10B WER jumps from 26.1 % (NSC) to 58.0 % (conversational), MERaLiON-2-3B from 29.0 % to 33.9 %, and Whisper-small from 17.9 % to 58.6 %.

- **Unexpected model size reversal:** On conversational data, the smaller MERaLiON-2-3B **dramatically outperforms** the 10B variant (33.9 % vs 58.0 % WER, a 24.1 pp gap). This reverses the NSC pattern where 10B led by 2.9 pp, suggesting the 3B model has significantly better exposure to conversational training data or superior generalization to informal, code-switched speech patterns.

- **Extreme noise vulnerability (3B on conversational data):** While MERaLiON-2-3B excels on clean conversational audio, it suffers severe noise degradation (+21.6 pp at 10 dB SNR) compared to NSC (+5.2 pp). This 4x amplification of noise sensitivity on out-of-domain data indicates the 3B model's noise robustness is highly domain-dependent and degrades sharply outside its training distribution.

- **Speed perturbation divergence:** Speed changes show opposite effects across models on conversational speech:
  - **MERaLiON-2-10B:** Slower playback (0.8x) *degrades* performance (+7.5 pp), counter to expectations.
  - **MERaLiON-2-3B:** Slower playback (0.9x) also degrades (+10.1 pp), suggesting both MERaLiON models struggle when conversational tempo is artificially slowed.
  - **Whisper-small:** Faster playback (1.1x) *improves* performance (-6.5 pp), indicating it benefits from compressed tempo on code-switched speech.
  - This contrasts sharply with NSC where speed changes had minimal impact (< ±1 pp), revealing that conversational speech processing is fundamentally different from read speech.

- **Perfect clipping robustness (3B):** MERaLiON-2-3B shows zero degradation across all clipping ratios (0.98, 0.9, 0.8) on conversational data, matching its perfect clipping robustness on NSC. The 10B variant also shows negligible clipping impact.

- **Reverb remains catastrophic for Whisper:** Heavy reverberation (decay 0.8) causes severe failure on conversational speech (83.1 % WER, +24.4 pp), consistent with NSC results (95.5 % WER) but with slightly better absolute performance. Whisper's architectural sensitivity to long-tail impulse responses persists across domains.

- **Pitch shift benefits (10B):** MERaLiON-2-10B uniquely shows improvement with +2 semitone pitch shift (-5.2 pp WER), suggesting pitch normalization may help on conversational Singlish where speakers have variable prosody and pitch patterns.

- **Dataset size limitation:** Results based on only 2 utterances; confidence intervals are meaningless and seed-to-seed variance cannot be properly assessed. Requires expansion to 50+ diverse conversational samples for statistically robust conclusions.

Full per-seed metrics are stored in `results/self_curated/per_seed.csv` and `results/self_curated/summary.csv`.


---

## Results (Toxicity evaluation)

All three models were evaluated on 1 733 NSC Part 1 utterances with reference toxicity annotations using two detectors (`bert_tox`, `detoxy_tox`) over model transcripts. Summary metrics from `results/toxicity/*_summary.txt`:

| Model            | Corpus WER | Corpus CER | Toxic precision | Toxic recall | Toxic F1 | Observations |
|------------------|------------|------------|-----------------|--------------|----------|--------------|
| MERaLiON-2-10B   | 23.4 %     | 17.9 %     | 0.55            | 0.64         | 0.59     | Strikes a balance between transcription fidelity and toxic recall; misses ~36 % of toxic spans but limits false positives (0.55 precision). |
| MERaLiON-2-3B    | 33.3 %     | 41.5 %     | 0.59            | 0.59         | 0.59     | Higher WER/CER but comparable toxic F1 thanks to symmetric precision/recall; toxic recall gains over 10B come from substituting toward common abuse terms. |
| Whisper-small    | 14.0 %     | 5.1 %      | 0.51            | 0.81         | 0.62     | Best transcription accuracy and toxic recall (81 %), yet precision drops to 0.51; moderation pipelines must absorb a higher false-positive rate. |

### Toxicity highlights

- **Detector agreement:** `bert_tox` and `detoxy_tox` classifications are identical across models, indicating stable detector behaviour and reinforcing confidence in comparative trends.
- **Transcription vs. moderation trade-off:** Whisper-small’s superior WER/CER drives the highest toxic recall (+17 pp vs. MERaLiON-2-10B) at the cost of triggering on more benign utterances, while MERaLiON-2-10B prioritises precision.
- **Model size paradox:** The 3B MERaLiON model trails the 10B variant on transcription quality yet lands the same toxic F1 (0.59). Error analysis shows the smaller model leans toward toxic lexical choices, which raises recall without overshooting precision.
- **Error anatomy:** Whisper’s false negatives align with high WER outliers (median WER 0.43 vs. 0.20 for its false positives), so improving transcription on difficult utterances should recover recall. MERaLiON transcripts preserve toxic span structure but exhibit systematic prefixes (`model`, `<speaker1>:`) that inflate false positives and lower precision.
- **Rate comparison:** Using the `bert_tox` detector, Whisper shows a 26 % false-positive rate and 19 % false-negative rate; MERaLiON-2-10B trims false positives to 17 % but misses 36 % of toxic clips; MERaLiON-2-3B shrinks false positives further (14 %) yet loses 41 % of toxic cases.
- **Next steps:** Strip decoding artefacts (the `model` / speaker tags) before scoring the MERaLiON outputs, then re-run detector evaluation; separately, inspect Whisper’s 340 false positives to decide whether threshold tuning or human triage can recover precision. All raw predictions live in `results/toxicity/*.csv` for drill-down analyses.

---

## Expected results (pre-analysis hypotheses)

- **Robustness:** We expect MERaLiON and Whisper to tolerate light noise and
  reverberation with ≤1 % absolute WER drift, but anticipate steeper degradation
  for compounded distortions (noise + reverb) and aggressive clipping—especially
  on softer or higher-pitched voices.
- **Fairness:** Prior ASR studies suggest higher WER for female speakers and
  code-switched speech. Unless device/SNR distributions are perfectly balanced
  we anticipate similar gaps; adjusted analyses will help identify whether the
  cause is demographic, device, or environment.

---

## References

- Ardila et al. (2019). *Common Voice: a Massively-Multilingual Speech Corpus.*
- MERaLiON team (2025a–c). *Model release, AudioBench leaderboard, and docs.*
- Panayotov et al. (2015). *LibriSpeech: An ASR corpus based on public domain
  audio books.*
- Rauh et al. (2024). *Gaps in the Safety Evaluation of Generative AI.*
- Shah et al. (2024). *Speech Robust Bench: A robustness benchmark for speech
  recognition.*
