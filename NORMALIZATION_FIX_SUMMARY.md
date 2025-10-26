# Normalization Fix Summary

## Issue Discovered

The WER/CER calculation was **biased against MERaLiON** models due to insufficient text normalization:

### Problems:
1. **MERaLiON outputs included formatting artifacts:**
   - `"model there were barrels..."` (model prefix)
   - `"model <speaker1>: it's my final day..."` (model + speaker tags)

2. **Whisper outputs were clean:**
   - `" there were barrels..."` (just leading space)

3. **Old normalization only did:**
   - ToLowerCase()
   - RemoveMultipleSpaces()
   - Strip()
   
4. **Missing:** Removal of "model" prefix and `<speaker#>` tags

## Fix Applied

Updated `asr_eval/metrics/wer_cer.py` to add:
```python
jiwer.SubstituteRegexes({
    r"<speaker\d+>:?\s*": "",  # Remove speaker tags
    r"^model\s*:?\s*": "",      # Remove model prefix
})
```

## Impact on Results

### Robustness Evaluation (NSC Part 1, 682 utterances)

**OLD Results (from README):**
- MERaLiON-2-10B: WER 26.1%, CER 22.1%
- MERaLiON-2-3B:  WER 29.0%, CER 29.3%
- Whisper-small:  WER 17.9%, CER 6.1%

**NEW Results (with fixed normalization):**
- MERaLiON-2-10B: WER **13.6%**, CER **3.3%** (↓12.5pp WER, ↓18.8pp CER!)
- MERaLiON-2-3B:  WER **13.1%**, CER **3.1%** (↓15.9pp WER, ↓26.2pp CER!)
- Whisper-small:  WER 17.9%, CER 6.1% (unchanged)

### Key Findings:

1. **MERaLiON models were significantly underestimated**
   - The "model" prefix and speaker tags inflated error rates by ~12-16pp WER
   - CER was even more affected (~19-26pp inflation)

2. **MERaLiON-2-10B now BEATS Whisper on clean NSC**
   - 13.6% vs 17.9% WER (4.3pp better!)
   - Previously reported as 8.2pp worse

3. **MERaLiON-2-3B nearly matches the 10B variant**
   - Only 0.5pp WER difference (13.1% vs 13.6%)
   - Despite being trained on 40% less data

## What Needs Re-Running?

### ✅ NO Re-run Needed:
- **Robustness evaluation** - The transcripts are already saved in `results/robustness/details.jsonl`
- **Self-curated evaluation** - Transcripts saved in `results/self_curated/details.jsonl`
- We can recalculate metrics from existing data

### ⚠️ Analysis Scripts Need Update:
- Any analysis scripts that calculated WER/CER from the saved results
- Tables and charts in the README
- Final report numbers

### ❌ May Need Re-run (needs investigation):
- **Fairness evaluation** - Has a separate issue where hypothesis includes full audio but reference is truncated to 30s
- **Toxicity evaluation** - Need to check if WER/CER are used there

## Next Steps

1. ✅ **Fix applied** to `asr_eval/metrics/wer_cer.py`
2. ⏭️ Recalculate all summary statistics from existing JSONL files
3. ⏭️ Update README.md with corrected numbers
4. ⏭️ Update any visualizations/charts
5. ⏭️ Update final report if it contains these numbers
6. ⏭️ Commit the fix with clear explanation
