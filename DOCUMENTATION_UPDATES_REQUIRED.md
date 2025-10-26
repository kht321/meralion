# Documentation Updates Required After Normalization Fix

## Summary of Changes

The normalization fix dramatically improved MERaLiON results. **MERaLiON now BEATS Whisper!**

### Key Metric Changes:

**NSC Part 1 (682 utterances):**
- MERaLiON-2-10B: 26.1% → **13.6% WER** (12.5pp improvement!)
- MERaLiON-2-3B:  29.0% → **13.1% WER** (15.9pp improvement!)
- Whisper-small: 17.9% WER (unchanged)

**Result:** MERaLiON-2-10B now outperforms Whisper by 4.3pp!

**Self-Curated Conversational (20 samples):**
- MERaLiON-2-10B: 39.8% → **66.5% WER**
- MERaLiON-2-3B:  23.8% → **38.8% WER**  
- Whisper-small: 38.0% → **52.2% WER**

---

## Files Requiring Updates

### 1. README.md ✅ PARTIALLY UPDATED

**Line 318-322:** NSC Part 1 Results Table
- ✅ Updated clean WER/CER values
- ✅ Updated worst corruption WER

**Line 326-328:** Baseline Accuracy Insight  
- ❌ STILL SAYS: "Whisper-small outperforms both MERaLiON variants"
- ✅ SHOULD SAY: "MERaLiON-2-10B outperforms Whisper by 4.3pp (13.6% vs 17.9%)"

**Line 330-336:** Model Size Impact
- ❌ STILL SAYS: "+2.9 pp WER degradation", "significantly higher CER (+7.2 pp)"
- ✅ SHOULD SAY: "+0.5 pp WER degradation", "minimal CER gap (3.1% vs 3.3%)"

**Line 338-344:** Noise Robustness
- ❌ STILL SAYS: "reaches 32.3% (+6.2 pp)", "despite higher clean error rates"
- ✅ SHOULD SAY: "reaches 19.4% (+5.8 pp)", "with both clean accuracy and robustness advantages"

**Line 370:** Accuracy vs Robustness Chart Caption
- ❌ STILL SAYS: "Whisper-small achieves best clean WER"
- ✅ SHOULD SAY: "MERaLiON achieves best clean WER; Whisper worst robustness"

**Line 384-388:** Self-Curated Table
- ❌ NOT UPDATED: Still shows old conversational WER values
- ✅ NEEDS: 10B: 66.5%, 3B: 38.8%, Whisper: 52.2%

**Line 390-399:** Conversational Observations
- ❌ Multiple WER values need updating throughout paragraphs

---

### 2. scripts/final_report.py (Final Report Generator)

**Line 68-76:** Abstract
```python
# OLD:
"Testing on 682 National Speech Corpus utterances reveals MERaLiON models demonstrate
exceptional robustness (+0.3-0.5pp average degradation under noise, speed, pitch, reverb)"

# NEW:
"Testing on 682 National Speech Corpus utterances reveals MERaLiON models demonstrate  
exceptional robustness (+0.5-0.6pp average degradation under noise, speed, pitch, reverb)
and superior baseline accuracy (13.6% WER vs Whisper's 17.9%)"
```

**Line 183-189:** Table 1 - Robustness Summary (NSC Part 1)
```python
rows = [
    # OLD:
    ["MERaLiON-2-10B", "26.1%", "22.1%", "+0.3pp", "+6.2pp", "Noise 10dB (32.3%)"],
    ["MERaLiON-2-3B", "29.0%", "29.3%", "+0.5pp", "+5.1pp", "Noise 10dB (34.1%)"],
    
    # NEW:
    ["MERaLiON-2-10B", "13.6%", "3.3%", "+0.6pp", "+5.8pp", "Noise 10dB (19.4%)"],
    ["MERaLiON-2-3B", "13.1%", "3.1%", "+0.5pp", "+5.1pp", "Noise 10dB (18.2%)"],
]
```

**Line 196-206:** Robustness Discussion Paragraph
```python
# Multiple WER/CER values need updating:
# - "17.9% WER vs. 26-29% for MERaLiON" → "17.9% WER vs. 13-14% for MERaLiON"
# - "Whisper-small achieves superior clean accuracy" → "MERaLiON achieves superior clean accuracy"
# - "+5.1pp vs. +6.2pp" → "+5.1pp vs. +5.8pp"
```

**Line 209-216:** Table 2 - Conversational Dataset  
```python
rows = [
    # OLD:
    ["MERaLiON-2-10B", "39.8%", "+0.8pp", "+5.3pp (Speed 0.8x)", "+13.7pp"],
    ["MERaLiON-2-3B", "23.8%", "+3.5pp", "+12.3pp (Noise 10dB)", "-5.2pp"],
    ["Whisper-small", "38.0%", "+3.6pp", "+23.8pp (Reverb 0.8)", "+20.1pp"],
    
    # NEW:
    ["MERaLiON-2-10B", "66.5%", "-8.5pp", "+0.4pp (Clipping 0.8)", "+52.9pp"],
    ["MERaLiON-2-3B", "38.8%", "+2.8pp", "+28.2pp (Noise 10dB)", "+25.7pp"],
    ["Whisper-small", "52.2%", "+6.4pp", "+38.5pp (Reverb 0.8)", "+34.3pp"],
]
```

**Line 219-227:** Conversational Discussion
```python
# Update multiple passages:
# - "16pp gap" → "27.7pp gap" (66.5% - 38.8%)
# - "3B dramatically outperforms 10B" → "10B now underperforms on conversational"
# - Domain shift values all need recalculation
```

**Line 386-394:** Table 6 - Fairness Analysis
```python
# Overall WER needs recalculation with fixed normalization
# Gender/Race disparities may change slightly
```

**Line 420-429:** Conclusion Paragraph
```python
# OLD:
"exceptional acoustic robustness (+0.3-0.5pp average degradation) versus Whisper's reverb 
catastrophe (+77.6pp), validating suitability for Singapore's urban environments."

# NEW:
"exceptional acoustic robustness (+0.5-0.6pp average degradation) AND superior baseline
accuracy (13.6% vs 17.9% WER), decisively outperforming Whisper despite catastrophic
reverb vulnerability (+77.6pp)."
```

---

### 3. docs/Final Report.docx (Generated Output)

**Sections Affected:**
1. Abstract (page 1)
2. Table 1 - Robustness Summary (page 4)
3. Robustness results discussion (page 4-5)
4. Table 2 - Conversational results (page 5)
5. Conversational discussion (page 5-6)
6. Conclusion (page 9)

**Action Required:**
- Re-run `python scripts/final_report.py` after updating the script
- This will regenerate `docs/Final Report.docx` with correct numbers

---

## Priority Order

1. **HIGH:** Update README.md baseline accuracy section (changes conclusion!)
2. **HIGH:** Update scripts/final_report.py Tables 1 & 2
3. **MEDIUM:** Update README.md noise robustness text
4. **MEDIUM:** Update scripts/final_report.py discussion paragraphs
5. **LOW:** Update chart captions in README.md
6. **AFTER:** Regenerate Final Report.docx

---

## Verification Checklist

After updates, verify:
- [ ] README table shows MERaLiON-2-10B: 13.6% WER
- [ ] README states "MERaLiON outperforms Whisper"
- [ ] Final report abstract mentions "+0.5-0.6pp" not "+0.3-0.5pp"
- [ ] Final report Table 1 shows 13.6% / 13.1% WER
- [ ] Final report Table 2 shows 66.5% / 38.8% / 52.2% WER
- [ ] Conclusion updated to reflect MERaLiON superiority
- [ ] docs/Final Report.docx regenerated with new script

---

## Impact Statement

**Before Fix:**
- "Whisper achieves better clean accuracy than MERaLiON"
- "MERaLiON has high WER but good robustness"

**After Fix:**
- "MERaLiON achieves 4.3pp better WER than Whisper AND exceptional robustness"
- "MERaLiON is the clear winner for Singapore ASR applications"

This changes the entire narrative of the evaluation! 🎉
