# Comprehensive Documentation Review - All Metrics

## Expected Values (from recalculated_metrics.json):

### NSC Part 1 (682 utterances):
- MERaLiON-2-10B: 13.6% WER, 3.3% CER, +0.6pp avg ΔWER, +5.8pp worst (19.4% at Noise 10dB)
- MERaLiON-2-3B:  13.1% WER, 3.1% CER, +0.5pp avg ΔWER, +5.1pp worst (18.2% at Noise 10dB)
- Whisper-small:  17.9% WER, 6.1% CER, +9.6pp avg ΔWER, +77.6pp worst (95.5% at Reverb 0.8)

### Self-Curated Conversational (20 samples):
- MERaLiON-2-10B: 66.5% WER, 53.0% CER, -8.5pp avg ΔWER, +0.4pp worst
- MERaLiON-2-3B:  38.8% WER, 29.7% CER, +2.8pp avg ΔWER, +28.2pp worst
- Whisper-small:  52.2% WER, 42.8% CER, +6.4pp avg ΔWER, +38.5pp worst

---

## Review Checklist:

### README.md:
- [ ] Line 318-322: NSC Part 1 table
- [ ] Line 326-330: Baseline accuracy insight
- [ ] Line 332-338: Model size impact insight
- [ ] Line 340-346: Noise robustness insight
- [ ] Line 370: Figure 2 caption
- [ ] Line 384-388: Self-curated table
- [ ] Line 391-395: Domain shift insight
- [ ] Line 397-401: Model size reversal insight
- [ ] Line 403-407: Noise vulnerability insight

### scripts/final_report.py:
- [ ] Line 68: Abstract robustness values
- [ ] Line 186-187: Table 1 NSC results
- [ ] Line 212-214: Table 2 Conversational results
- [ ] Line 220-224: Conversational discussion
- [ ] Line 422-424: Conclusion

### docs/Final Report.docx:
- [ ] Page 1: Abstract
- [ ] Page 4: Table 1
- [ ] Page 5: Table 2
- [ ] Page 9: Conclusion
