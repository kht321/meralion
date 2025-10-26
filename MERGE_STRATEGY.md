# Branch Merge Strategy

## Current Situation

**Your local main:** 1 commit ahead (normalization fix)
**Remote branches to merge:**
1. `origin/zexel` - Guardrail metric fixes and transcriber improvements
2. `origin/fairness` - Fairness evaluation results and code cleanup

## Branch Analysis

### 1. origin/zexel (Zexel's work)

**Key Changes:**
- ✅ **Guardrail metric fixes** (`run_guardrail_eval.py`) - IMPORTANT!
  - Fixed pass_through_rate calculation to use `final_text` instead of `raw`
  - Fixed false_block_rate to detect `[censored]` markers
  - Added proper text normalization for metric calculations
  
- ✅ **Transcriber fixes** (`asr_eval/models/meralion.py`)
  - Reverted TRANSCRIBE_PROMPT back to simple "Please transcribe this speech."
  - Was previously "Replace profanities with [CENSOREDTEXT]" which biased results

- ⚠️ **Does NOT include WER/CER normalization fix**
  - Still has old version of `wer_cer.py`

**Conflicts:** None with normalization fix (different files)

### 2. origin/fairness (Fairness evaluation work)

**Key Changes:**
- ✅ **Fairness evaluation results** - New evaluation data
- ✅ **Code cleanup** (`asr_eval/models/meralion.py`)
  - Removed logging statements
  - Removed NO_TOXIC_TRANSCRIBE_PROMPT and TOXICITY_CLASSIFY_PROMPT
  - Removed `no_toxic_transcribe()` and `classify_toxicity()` methods
  
- ⚠️ **Large diff:** 403+ line changes to README, deleted DOCKER.md/LICENSE/etc
- ⚠️ **Does NOT include WER/CER normalization fix**

**Conflicts:** 
- Likely conflicts in `asr_eval/models/meralion.py` (both branches modify it)
- May conflict with LICENSE and .gitignore changes from main

### 3. Your main (Normalization fix + License)

**Key Changes:**
- ✅ **Critical WER/CER normalization fix** in `wer_cer.py`
- ✅ **Added MIT LICENSE**
- ✅ **Updated .gitignore** to exclude docs/

## Recommended Merge Order

### Option 1: Cherry-pick Important Fixes (RECOMMENDED)

This avoids massive fairness branch changes while getting critical fixes:

```bash
# 1. Cherry-pick zexel's guardrail metric fixes
git cherry-pick d2d47cf  # Fix compute metrics
git cherry-pick f82a2c6  # Fixes to transcriber (revert prompt)

# 2. Selectively merge fairness evaluation data only
git checkout origin/fairness -- results/fairness/
git checkout origin/fairness -- configs/fairness.yaml
git checkout origin/fairness -- data/manifests/fairness_manifest.jsonl
git commit -m "Add fairness evaluation results from fairness branch"

# 3. Push consolidated changes
git push origin main
```

### Option 2: Full Merge (More Complex)

```bash
# 1. Merge zexel first (cleaner)
git merge origin/zexel --no-ff -m "Merge zexel branch: guardrail metrics and transcriber fixes"

# 2. Merge fairness (will have conflicts)
git merge origin/fairness --no-ff -m "Merge fairness branch: evaluation results"
# Resolve conflicts keeping:
# - Your LICENSE
# - Your .gitignore
# - Your wer_cer.py normalization fix
# - Their fairness results
```

## Critical Files to Preserve

**MUST KEEP YOUR VERSION:**
- `asr_eval/metrics/wer_cer.py` - Your normalization fix
- `LICENSE` - Your MIT license
- `.gitignore` - Your docs exclusion

**CAN MERGE:**
- `asr_eval/whitebox/run_guardrail_eval.py` - Zexel's metric fixes
- `asr_eval/models/meralion.py` - Need to merge carefully (both branches touch it)
- `results/fairness/*` - Fairness evaluation data

## Conflict Resolution Guide

If `asr_eval/models/meralion.py` conflicts:
- Keep zexel's TRANSCRIBE_PROMPT reversion (simple prompt)
- Keep fairness branch's logging removal
- Keep main's toxicity methods (if any conflicts)

## Recommendation

**Use Option 1 (Cherry-pick)** because:
1. Cleaner history
2. Avoid massive fairness branch changes (LICENSE deletion, DOCKER.md deletion)
3. Get all critical fixes quickly
4. Less conflict resolution needed

After merging, you'll need to:
1. Recalculate all metrics with fixed normalization
2. Update README with corrected results
3. Re-run fairness evaluation (it has the 30s truncation issue)
