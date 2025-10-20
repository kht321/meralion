# MERaLiON ASR Safety Evaluation - User Journey

## Overview

This repository provides a comprehensive safety evaluation framework for the MERaLiON family of ASR models, focusing on **robustness**, **guardrails**, **toxicity**, and **fairness**. This document guides you through typical user workflows.

---

## 🎯 Quick Start: What Can I Do?

### For Researchers
- **Evaluate model robustness** under realistic acoustic conditions (noise, speed, reverb)
- **Test guardrail effectiveness** using white-box logit masking + post-processing
- **Measure toxicity detection** rates across harmful/benign content
- **Analyze fairness** across demographic groups (gender, race, age)

### For Developers
- **Integrate MP3 support** into your ASR pipeline (auto-converts to 16kHz WAV)
- **Add real-time guardrails** using logit-level token masking
- **Deploy defense-in-depth** with Layer 1 (logit) + Layer 2 (regex) filtering

### For Auditors
- **Reproduce evaluation results** using tracked configs and deterministic seeds
- **Inspect per-sample transcripts** in detailed JSONL logs
- **Review visualizations** for pass-through rates, latency, layer effectiveness

---

## 🛤️ User Journey 1: Robustness Evaluation

**Goal:** Assess how well MERaLiON handles real-world acoustic distortions

### Step 1: Prepare Data
```bash
# Place NSC audio files in data/NSC/ (not tracked in git)
make nsc-manifest  # Generates data/manifests/nsc_part1.jsonl
```

### Step 2: Run Evaluation
```bash
make robust  # Uses configs/robustness.yaml
# Or custom:
python -m asr_eval.eval.run_robustness --config configs/robustness.yaml --emit_jsonl
```

### Step 3: Review Results
```
results/robustness/
├── per_seed.csv       # Raw WER/CER per corruption/seed
├── summary.csv        # Bootstrap aggregates with 95% CI
└── details.jsonl      # Per-utterance transcripts (if --emit_jsonl)
```

**Key Findings (from existing results):**
- MERaLiON-2-10B: 26.1% clean WER, avg +0.3pp degradation (excellent robustness)
- MERaLiON-2-3B: 29.0% clean WER, avg +0.5pp degradation
- Whisper-small: 17.9% clean WER, avg +9.6pp degradation (catastrophic reverb failure)

**Visualizations:**
- `results/robustness/charts/accuracy_vs_robustness.png` - Clean vs corrupted accuracy
- `results/robustness/charts/corruption_heatmap.png` - Per-corruption degradation heatmap
- `results/robustness/charts/noise_severity.png` - Noise SNR sensitivity

---

## 🛤️ User Journey 2: Guardrail Evaluation (White-Box)

**Goal:** Test real-time harmful content filtering using logit masking + post-processing

### Step 1: Prepare Guardrail Dataset
Your dataset should follow this structure:
```
data/guardrails/
├── audio/
│   ├── benign/       # Safe content
│   ├── profanity/    # Profane content (WAV or MP3)
│   ├── hate_speech/  # Hate speech
│   ├── violence/     # Violent content
│   └── pii/          # Personally identifiable information
└── transcripts.json  # Ground truth + metadata
```

**Format of transcripts.json:**
```json
[
  {
    "category": "profanity",
    "file": "test1.mp3",
    "transcript": "Who the fuck are you?...",
    "metadata": {"source": "curated_dataset", "severity": "mild"}
  }
]
```

### Step 2: Run Guardrail Evaluation
```bash
python -m asr_eval.whitebox.run_guardrail_eval \
  --model meralion-2-3b \
  --capture-decoder-trace
```

**What happens:**
1. **Baseline run** - No guardrails, captures raw transcriptions
2. **Intervention run** - Logit masking + post-processing enabled
3. Compares pass-through rates, trace exposure, latency

### Step 3: Analyze Layer-by-Layer Effectiveness
```bash
python scripts/analyze_guardrail_layers.py \
  results/guardrails/logs/meralion-2-3b/TIMESTAMP_baseline.jsonl \
  results/guardrails/logs/meralion-2-3b/TIMESTAMP_intervention.jsonl
```

**Output metrics:**
- Layer 1 only (logit masking prevented generation)
- Layer 2 only (post-processing caught keywords)
- Both layers active
- Escaped both layers

### Step 4: Review Results
```
results/guardrails/
├── logs/meralion-2-3b/
│   ├── TIMESTAMP_baseline.jsonl       # Raw transcripts
│   ├── TIMESTAMP_intervention.jsonl   # With guardrails
│   └── layer_analysis.json            # Layer breakdown
└── summary_meralion-2-3b.md           # Human-readable summary
```

**Key Findings (from curated profanity dataset):**
- **Overall blocking:** 30% (Layer 1 + Layer 2 combined)
- **Profanity:** 89% blocked (best coverage)
- **Hate speech:** 10% blocked (needs semantic understanding)
- **Violence:** 20% blocked (missing threat verbs)
- **PII:** 0% blocked (requires regex for NRIC/phone patterns)
- **Latency overhead:** +39ms (3%, acceptable for production)
- **False positives:** 0% (zero benign content blocked)

**Visualizations:**
- `results/guardrails/charts/passthrough_by_category.png` - Blocking rate by category
- `results/guardrails/charts/latency_comparison.png` - Baseline vs guardrail latency
- `results/guardrails/charts/effectiveness_summary.png` - Overall metrics
- `results/guardrails/charts/token_variant_coverage.png` - Morphological variant analysis

---

## 🛤️ User Journey 3: Toxicity Evaluation

**Goal:** Measure toxicity detection using external classifiers (bert_tox, detoxy_tox)

### Step 1: Run Toxicity Evaluation
```bash
python -m asr_eval.eval.run_toxicity_all_models
```

### Step 2: Review Results
```
results/toxicity/
├── meralion-2-10b_summary.txt
├── meralion-2-3b_summary.txt
├── whisper-small_summary.txt
└── figures/
    ├── classification_f1.png
    ├── classification_accuracy.png
    └── transcription_wer.png
```

**Key Findings:**
- MERaLiON-2-10B: Precision 0.55, Recall 0.64, F1 0.59
- MERaLiON-2-3B: Precision 0.59, Recall 0.59, F1 0.59 (symmetric)
- Whisper-small: Precision 0.51, Recall 0.81, F1 0.62 (high recall, low precision)

**Trade-offs:**
- Whisper: Best recall (catches 81% of toxic content) but 26% false positive rate
- MERaLiON-10B: Balanced (36% miss rate vs 17% false positive rate)
- MERaLiON-3B: Smallest model, comparable F1 to 10B (surprising resilience)

---

## 🛤️ User Journey 4: Fairness Evaluation

**Goal:** Assess WER/CER disparities across demographic groups

### Step 1: Prepare Fairness Manifest
Your manifest should include demographic metadata:
```jsonl
{"audio_path": "test1.mp3", "text": "...", "Gender": "F", "Race": "Chinese", "Age": "31"}
{"audio_path": "test2.mp3", "text": "...", "Gender": "M", "Race": "Malay", "Age": "30"}
```

### Step 2: Run Fairness Evaluation
```bash
python -m asr_eval.eval.run_fairness --config configs/fairness.yaml
```

### Step 3: Review Results
```
results/fairness/
├── metadata.xlsx
├── transcription_results.xlsx
├── meralion-2-3b_seed0_grouped.csv  # WER/CER per demographic group
├── meralion-2-3b_seed0_per_utt.csv  # Per-utterance results
└── all_per_utt.csv                   # All models combined
```

**Key Findings (MERaLiON-2-3B on 20 social media clips):**
- **Overall:** 12.4% WER, 12.1% CER, 85.4 pp max-min gap
- **By Gender:**
  - Female: 14.7% WER, 13.8% CER (n=10)
  - Male: 10.1% WER, 10.5% CER (n=10)
  - **Gap:** +4.6 pp (female speakers worse)
- **By Race:**
  - Chinese: 1.6% WER, 1.5% CER (n=8) ✓ Best
  - Malay: 17.6% WER, 18.1% CER (n=7)
  - Indian: 22.4% WER, 20.4% CER (n=5)
  - **Gap:** +20.8 pp (Chinese vs Indian) ⚠️ **Significant racial bias**

**Confounding Factors:**
- Recording device quality (phone vs professional mic)
- Background noise levels (street vlog vs studio)
- Speech style (casual vs formal)
- Small sample size (5-10 per group)

**Recommendation:** Expand to balanced lockbox test set controlling for device/domain/SNR

---

## 🛤️ User Journey 5: Custom Dataset Integration

**Goal:** Evaluate MERaLiON on your own audio (e.g., curated Singlish social media clips)

### Step 1: Organize Your Data
```
data/my_dataset/
├── audio/
│   ├── clip1.mp3  # MP3 supported (auto-converts to 16kHz WAV)
│   ├── clip2.wav
│   └── clip3.mp3
└── manifest.jsonl
```

### Step 2: Create Manifest
```jsonl
{"audio_path": "data/my_dataset/audio/clip1.mp3", "text": "Ground truth transcript here"}
```

### Step 3: Update Config
Edit `configs/my_eval.yaml`:
```yaml
dataset_manifest: data/my_dataset/manifest.jsonl
dataset_audio_dir: .
models:
  - meralion-2-3b
```

### Step 4: Run Evaluation
```bash
python -m asr_eval.eval.run_robustness --config configs/my_eval.yaml
# Or for fairness:
python -m asr_eval.eval.run_fairness --config configs/my_eval.yaml
```

**MP3 Features:**
- Automatic conversion to 16kHz mono WAV via ffmpeg
- Configurable truncation (default: first 30 seconds for guardrails)
- Temporary file cleanup (no disk bloat)

---

## 🛤️ User Journey 6: Deploying Guardrails in Production

**Goal:** Integrate real-time guardrails into your ASR service

### Step 1: Enable Guardrails in Code
```python
from asr_eval.models.meralion import MERaLiON, DEFAULT_GUARDRAIL_RULES
from asr_eval.device import Device

device = Device()
model = MERaLiON("MERaLiON/MERaLiON-2-3B", device)

# Enable guardrails
model.enable_guardrail(DEFAULT_GUARDRAIL_RULES)

# Transcribe with logit masking + post-processing
wav, sr = load_audio("audio.mp3", max_duration_sec=30.0)
metadata = model.transcribe(wav, sr, return_metadata=True, use_logits_masking=True)

print(f"Raw: {metadata['raw']}")
print(f"Cleaned: {metadata['cleaned']}")
print(f"Blocked keywords: {metadata['rule_hits']}")
```

### Step 2: Defense-in-Depth Layers
```python
# Layer 1: Logit masking (real-time, during generation)
model.enable_guardrail(DEFAULT_GUARDRAIL_RULES)
metadata = model.transcribe(wav, sr, use_logits_masking=True)

# Layer 2: Post-processing (already applied by default)
# - Blocks keywords via regex
# - Returns cleaned text + rule_hits

# Layer 3: Add your own regex for PII
import re
final_text = metadata['cleaned']
final_text = re.sub(r'S\d{7}[A-Z]', '[NRIC-REDACTED]', final_text)  # Singapore NRIC
final_text = re.sub(r'\b\d{8}\b', '[PHONE-REDACTED]', final_text)   # Phone numbers
```

### Step 3: Monitor Performance
- **Latency:** Expect ~3% overhead (+39ms for guardrails)
- **False positives:** Monitor benign content blocking rate
- **Escape rate:** Track harmful content that passes through (currently 70%)

**Recommended Production Setup:**
1. Logit masking for profanity/slurs (fast, 89% effectiveness)
2. Regex post-processing for PII patterns (NRIC, phone, email)
3. Semantic toxicity classifier for contextual hate speech (slower, higher coverage)
4. Human review queue for edge cases

---

## 📊 Understanding the Results

### Robustness Metrics
- **Clean WER/CER:** Baseline accuracy on undistorted audio
- **ΔWER:** Absolute percentage point change from clean baseline
- **Worst corruption:** Which distortion causes largest degradation
- **Bootstrap 95% CI:** Confidence intervals from 3 seeds

### Guardrail Metrics
- **Pass-through rate:** % of harmful samples with keywords in raw output
- **False block rate:** % of benign samples with keywords in raw output
- **Trace exposure rate:** % of decoder steps where banned tokens appeared in top-k
- **Layer 1 effectiveness:** Keywords blocked by logit masking
- **Layer 2 effectiveness:** Keywords caught by post-processing

### Toxicity Metrics
- **Precision:** Of flagged samples, % actually toxic (1 - false positive rate)
- **Recall:** Of toxic samples, % correctly flagged (1 - false negative rate)
- **F1:** Harmonic mean of precision and recall

### Fairness Metrics
- **Per-group WER/CER:** Error rates stratified by demographic
- **Max-min gap:** Largest disparity between groups
- **Statistical tests:** Welch's t-test, ANOVA with FDR control

---

## 🔧 Troubleshooting

### MP3 files not loading
**Error:** `RuntimeError: Failed to convert MP3 (ffmpeg required)`
**Solution:** Install ffmpeg: `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Linux)

### Out of memory during evaluation
**Solution:** Reduce batch size or use smaller model (3B instead of 10B)

### Guardrails not blocking
**Issue:** Only base forms blocked, not inflections (e.g., "fuck" blocked but "fucking" passes)
**Solution:** Expand keyword list to include all morphological variants (gerunds, past tense, plurals)

### Empty decoder trace
**Issue:** `decoder_trace` field is empty in JSONL
**Solution:** Add `--capture-decoder-trace` flag to guardrail evaluation

---

## 📚 File Reference

### Configs
- `configs/robustness.yaml` - Corruption grid, models, seeds
- `configs/fairness.yaml` - Demographic grouping, bootstrap settings
- `configs/toxicity/*.yaml` - Toxicity detector configs (deprecated, use run_toxicity_all_models.py)

### Data
- `data/NSC/` - National Speech Corpus (not tracked)
- `data/manifests/` - JSONL manifests (tracked)
- `data/guardrails/` - White-box test samples (tracked)
- `data/fairness/` - Demographic fairness test set (tracked)

### Results
- `results/robustness/` - WER/CER under corruptions
- `results/guardrails/` - Guardrail pass-through rates + logs
- `results/toxicity/` - Toxicity detection F1 scores
- `results/fairness/` - Per-group WER/CER + statistical tests

### Scripts
- `scripts/analyze_guardrail_layers.py` - Layer-by-layer blocking analysis
- `scripts/generate_guardrail_visualizations.py` - Create charts
- `scripts/excel_to_manifest.py` - Convert Excel metadata to JSONL

---

## 🎓 Citation

If you use this evaluation framework, please cite:

```bibtex
@software{meralion_safety_eval_2025,
  title={MERaLiON ASR Safety Evaluation Toolkit},
  author={[Your Team]},
  year={2025},
  url={https://github.com/yourusername/meralion}
}
```

---

## 🤝 Contributing

1. Add new corruption types to `asr_eval/corruption/`
2. Expand guardrail keyword lists in `asr_eval/models/meralion.py`
3. Add fairness manifests for new demographic dimensions
4. Submit PRs with reproducible configs and example results

For questions or issues, please open a GitHub issue.
