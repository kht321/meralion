#!/usr/bin/env python3
"""Script to update README.md and Final Report with corrected metrics."""

import json
from pathlib import Path

# Load recalculated metrics
with open("results/recalculated_metrics.json") as f:
    metrics = json.load(f)

print("=" * 70)
print("UPDATED METRICS FOR DOCUMENTATION")
print("=" * 70)

robust = metrics['robustness']
curated = metrics['self_curated']

print("\n📊 NSC Part 1 Robustness (README Table Rows):")
print("-" * 70)
print("| MERaLiON-2-10B   | 13.6 %    | 3.3 %     | +0.6 pp  | +5.8 pp    | Noise SNR 10 dB (19.4 %) |")
print("| MERaLiON-2-3B    | 13.1 %    | 3.1 %     | +0.5 pp  | +5.1 pp    | Noise SNR 10 dB (18.2 %) |")
print("| Whisper-small    | 17.9 %    | 6.1 %     | +9.6 pp  | +77.6 pp   | Reverb decay 0.8 (95.5 %) |")

print("\n📊 Self-Curated Conversational (README Table Rows):")
print("-" * 70)
print("| MERaLiON-2-10B   | 66.5 %    | 53.0 %    | -8.5 pp  | +0.4 pp    | Clipping 0.8 (66.9 %) |")
print("| MERaLiON-2-3B    | 38.8 %    | 29.7 %    | +2.8 pp  | +28.2 pp   | Noise SNR 10 dB (67.0 %) |")
print("| Whisper-small    | 52.2 %    | 42.8 %    | +6.4 pp  | +38.5 pp   | Reverb decay 0.8 (90.6 %) |")

print("\n📝 Key Changes for Documentation:")
print("-" * 70)
print("\n1. **README.md - NSC Part 1 Results Table** (Line ~318-322)")
print("   CHANGE:")
print("   - MERaLiON-2-10B: 26.1% → 13.6% WER, 22.1% → 3.3% CER")  
print("   - MERaLiON-2-3B:  29.0% → 13.1% WER, 29.3% → 3.1% CER")
print("   - Worst corruption: 32.3% → 19.4% for 10B, 34.1% → 18.2% for 3B")

print("\n2. **README.md - Baseline Accuracy Insight** (Line ~326-328)")
print("   OLD: 'Whisper-small outperforms both MERaLiON variants'")
print("   NEW: 'MERaLiON-2-10B outperforms Whisper by 4.3pp (13.6% vs 17.9%)'")

print("\n3. **README.md - Model Size Impact** (Line ~330-336)")
print("   CHANGE:")
print("   - WER gap: 2.9pp → 0.5pp (13.1% vs 13.6%)")
print("   - CER: No longer 'significantly higher', now 3.1% vs 3.3%")

print("\n4. **README.md - Noise Robustness** (Line ~338-344)")
print("   CHANGE:")
print("   - 10B at 10dB: 32.3% → 19.4%")
print("   - 3B at 10dB: 34.2% → 18.2%")
print("   - Update: 'despite higher clean error rates' → 'with both clean accuracy and robustness advantages'")

print("\n5. **README.md - Self-Curated Table** (Line ~384-388)")
print("   CHANGE:")
print("   - MERaLiON-2-10B: 39.8% → 66.5% WER")
print("   - MERaLiON-2-3B:  23.8% → 38.8% WER")
print("   - Whisper-small:  38.0% → 52.2% WER")

print("\n6. **Final Report - Abstract** (scripts/final_report.py Line ~68-70)")
print("   CHANGE:")
print("   - MERaLiON WER from 26% → 13.6%")
print("   - Update '+0.3-0.5pp' → '+0.5-0.6pp'")

print("\n7. **Final Report - NSC Results Table** (scripts/final_report.py Line ~183-189)")
print("   CHANGE:")
print("   - MERaLiON-2-10B: [26.1%, 22.1%, +0.3pp] → [13.6%, 3.3%, +0.6pp]")
print("   - MERaLiON-2-3B:  [29.0%, 29.3%, +0.5pp] → [13.1%, 3.1%, +0.5pp]")
print("   - Worst WER: 32.3% → 19.4% for 10B, 34.1% → 18.2% for 3B")

print("\n8. **Final Report - Conversational Table** (scripts/final_report.py Line ~209-216)")
print("   CHANGE:")
print("   - MERaLiON-2-10B: 39.8% → 66.5% WER")
print("   - MERaLiON-2-3B:  23.8% → 38.8% WER")
print("   - Whisper-small:  38.0% → 52.2% WER")

print("\n9. **Final Report - Discussion Sections**")
print("   - Update all WER/CER numbers in paragraphs")
print("   - Change conclusions from 'Whisper beats MERaLiON' to 'MERaLiON beats Whisper'")
print("   - Update domain shift gaps (e.g., '+13.7pp' for 10B NSC→Conv)")

print("\n" + "=" * 70)
print("✅ Use this reference to update documentation manually")
print("=" * 70)
