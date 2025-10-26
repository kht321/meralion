#!/usr/bin/env python3
"""Update README.md self-curated conversational section with corrected metrics."""

from pathlib import Path

readme_path = Path("README.md")
content = readme_path.read_text()

# Update 1: Self-curated table
old_table = """| Model            | Clean WER | Clean CER | Avg ΔWER | Worst ΔWER | Worst corruption (WER) | Best improvement (ΔWER) | Observations |
|------------------|-----------|-----------|----------|------------|------------------------|-------------------------|--------------|
| MERaLiON-2-10B   | 39.8 %    | 21.2 %    | +0.8 pp  | +5.3 pp   | Speed 0.8x (45.1 %)   | -1.5 pp (Speed 1.1x) | Excellent overall robustness on conversational speech (avg +0.8 pp); slowed playback degrades while faster playback helps. |
| MERaLiON-2-3B    | 23.8 %    | 15.4 %    | +3.5 pp  | +12.3 pp  | Noise SNR 10 dB (36.0 %) | -0.5 pp (Speed 1.1x) | Dramatically outperforms 10B by 16.0 pp on clean; moderate average degradation (+3.5 pp) but vulnerable to noise on conversational data. |
| Whisper-small    | 38.0 %    | 30.7 %    | +3.6 pp  | +23.8 pp  | Reverb decay 0.8 (61.7 %) | -3.8 pp (Speed 1.1x) | Poor baseline on conversational Singlish; average degradation (+3.6 pp) similar to 3B despite catastrophic reverb failure; faster playback improves performance. |"""

new_table = """| Model            | Clean WER | Clean CER | Avg ΔWER | Worst ΔWER | Worst corruption (WER) | Best improvement (ΔWER) | Observations |
|------------------|-----------|-----------|----------|------------|------------------------|-------------------------|--------------|
| MERaLiON-2-10B   | 66.5 %    | 53.0 %    | -8.5 pp  | +0.4 pp   | Clipping 0.8 (66.9 %) | -20.4 pp (Speed 1.2x) | Poor baseline on conversational speech; surprising negative avg ΔWER suggests corruptions sometimes help; large domain shift from NSC. |
| MERaLiON-2-3B    | 38.8 %    | 29.7 %    | +2.8 pp  | +28.2 pp  | Noise SNR 10 dB (67.0 %) | -2.0 pp (Pitch +2) | Best conversational performance; moderate average degradation (+2.8 pp) but extreme noise vulnerability on conversational data. |
| Whisper-small    | 52.2 %    | 42.8 %    | +6.4 pp  | +38.5 pp  | Reverb decay 0.8 (90.6 %) | -1.6 pp (Speed 1.1x) | Moderate baseline on conversational Singlish; catastrophic reverb failure; large domain shift from NSC (+34.3 pp). |"""

content = content.replace(old_table, new_table)

# Update 2: Domain shift impact
old_domain = """- **Domain shift impact:** All models perform worse on conversational Singlish compared to NSC read speech, though the gap is smaller than initially observed with 2 samples. MERaLiON-2-10B WER increases from 26.1 % (NSC) to 39.8 % (conversational, +13.7 pp), MERaLiON-2-3B from 29.0 % to 23.8 % (conversational is *better* by 5.2 pp), and Whisper-small from 17.9 % to 38.0 % (+20.1 pp). The 20-sample dataset reveals MERaLiON-2-3B actually excels on conversational data despite struggling on formal read speech."""

new_domain = """- **Domain shift impact:** All models exhibit significant performance degradation on conversational Singlish compared to NSC read speech. MERaLiON-2-10B WER increases dramatically from 13.6 % (NSC) to 66.5 % (conversational, +52.9 pp), MERaLiON-2-3B from 13.1 % to 38.8 % (+25.7 pp), and Whisper-small from 17.9 % to 52.2 % (+34.3 pp). The 20-sample dataset reveals severe domain mismatch, with MERaLiON-2-3B maintaining the best conversational performance despite all models struggling with informal, code-switched speech."""

content = content.replace(old_domain, new_domain)

# Update 3: Model size reversal
old_reversal = """- **Model size reversal confirmed:** On conversational data, the smaller MERaLiON-2-3B **dramatically outperforms** the 10B variant (23.8 % vs 39.8 % WER, a 16.0 pp gap). This reverses the NSC pattern where 10B led by 2.9 pp, suggesting the 3B model has significantly better exposure to conversational training data or superior generalization to informal, code-switched speech patterns. The expanded 20-sample dataset confirms this wasn't an artifact of small sample size—the 3B model is genuinely superior for conversational Singlish."""

new_reversal = """- **Model size reversal confirmed:** On conversational data, the smaller MERaLiON-2-3B **dramatically outperforms** the 10B variant (38.8 % vs 66.5 % WER, a 27.7 pp gap). This reverses the NSC pattern where models were nearly tied (0.5 pp difference), suggesting the 3B model has significantly better exposure to conversational training data or superior generalization to informal, code-switched speech patterns. The expanded 20-sample dataset confirms the 3B model is genuinely superior for conversational Singlish."""

content = content.replace(old_reversal, new_reversal)

# Update 4: Noise vulnerability
old_vuln = """- **Noise vulnerability (3B on conversational data):** While MERaLiON-2-3B excels on clean conversational audio, it suffers significant noise degradation (+12.3 pp at 10 dB SNR) compared to NSC (+5.2 pp). This 2.4× amplification of noise sensitivity on conversational data indicates the 3B model's noise robustness is domain-dependent, though less severe than the 4× amplification observed in the 2-sample pilot."""

new_vuln = """- **Noise vulnerability (3B on conversational data):** While MERaLiON-2-3B excels on clean conversational audio, it suffers extreme noise degradation (+28.2 pp at 10 dB SNR, reaching 67.0 % WER) compared to NSC (+5.1 pp). This 5.5× amplification of noise sensitivity on conversational data indicates the 3B model's noise robustness is severely domain-dependent, with conversational speech being particularly vulnerable."""

content = content.replace(old_vuln, new_vuln)

# Write updated content
readme_path.write_text(content)

print("✅ README.md conversational section updated successfully!")
print("\nUpdated sections:")
print("  1. Self-curated conversational table")
print("  2. Domain shift impact")
print("  3. Model size reversal")
print("  4. Noise vulnerability")
