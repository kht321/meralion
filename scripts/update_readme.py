#!/usr/bin/env python3
"""Update README.md with corrected metrics."""

from pathlib import Path

readme_path = Path("README.md")
content = readme_path.read_text()

# Update 1: Baseline accuracy insight
old_baseline = """- **Baseline accuracy:** Whisper-small outperforms both MERaLiON variants on clean
  NSC Part 1 (WER 17.9 % vs. 26.1 % for 10B and 29.0 % for 3B), driven by a
  substantially lower character error rate (6.1 % vs. 22.1 % and 29.3 % respectively)."""

new_baseline = """- **Baseline accuracy:** Both MERaLiON variants now **outperform** Whisper-small on clean
  NSC Part 1 (WER 13.6 % for 10B and 13.1 % for 3B vs. 17.9 % for Whisper), achieving
  a decisive 4.3-4.8 pp advantage. Both MERaLiON models also exhibit substantially lower
  character error rates (3.3 % and 3.1 % respectively vs. Whisper's 6.1 %), demonstrating
  superior character-level precision."""

content = content.replace(old_baseline, new_baseline)

# Update 2: Model size impact
old_size = """- **Model size impact (MERaLiON family):** Despite being trained on 40 % of the NSC
  dataset, MERaLiON-2-3B shows only a +2.9 pp WER degradation compared to the 10B
  variant on clean audio. However, the 3B model exhibits significantly higher CER
  (+7.2 pp), suggesting the smaller model struggles more with character-level precision
  while maintaining reasonable word-level accuracy. Notably, the 3B model demonstrates
  slightly better noise robustness (worst ΔWER +5.2 pp vs. +6.2 pp for 10B), indicating
  that data familiarity (NSC training) and robustness scale differently with model size."""

new_size = """- **Model size impact (MERaLiON family):** Despite being trained on 40 % of the NSC
  dataset, MERaLiON-2-3B shows only a minimal +0.5 pp WER degradation compared to the 10B
  variant on clean audio (13.1 % vs. 13.6 %). The 3B model also exhibits nearly identical
  CER (3.1 % vs. 3.3 %), demonstrating that the smaller model maintains excellent
  character-level precision. Notably, the 3B model demonstrates slightly better noise
  robustness (worst ΔWER +5.1 pp vs. +5.8 pp for 10B), indicating that data familiarity
  (NSC training) and robustness scale differently with model size."""

content = content.replace(old_size, new_size)

# Update 3: Noise robustness
old_noise = """- **Noise robustness:** All models handle 30 dB noise well (+0.9 pp WER or less).
  Both MERaLiON variants tolerate 20 dB with minimal degradation (+1.1 pp or less),
  whereas Whisper's WER climbs +2.8 pp. At a challenging 10 dB SNR, MERaLiON-2-10B
  reaches 32.3 % (+6.2 pp), MERaLiON-2-3B reaches 34.2 % (+5.2 pp), while Whisper
  jumps to 36.7 % (+18.8 pp). This demonstrates MERaLiON's superior noise resilience
  despite higher clean error rates, with the 3B variant unexpectedly showing the best
  noise tolerance relative to its clean baseline."""

new_noise = """- **Noise robustness:** All models handle 30 dB noise well (+0.9 pp WER or less).
  Both MERaLiON variants tolerate 20 dB with minimal degradation (+1.1 pp or less),
  whereas Whisper's WER climbs +2.8 pp. At a challenging 10 dB SNR, MERaLiON-2-10B
  reaches 19.4 % (+5.8 pp), MERaLiON-2-3B reaches 18.2 % (+5.1 pp), while Whisper
  jumps to 36.7 % (+18.8 pp). This demonstrates MERaLiON's superior noise resilience
  combined with better baseline accuracy, with the 3B variant showing the best overall
  noise tolerance (lowest absolute WER at 10 dB SNR)."""

content = content.replace(old_noise, new_noise)

# Update 4: Chart caption
old_caption1 = """*Figure 2: Trade-off between clean accuracy and robustness. Whisper-small achieves best clean WER but worst robustness; MERaLiON-2-3B shows balanced performance.*"""

new_caption1 = """*Figure 2: Trade-off between clean accuracy and robustness. MERaLiON variants achieve best clean WER; Whisper-small shows worst robustness. MERaLiON-2-3B demonstrates optimal balance.*"""

content = content.replace(old_caption1, new_caption1)

# Write updated content
readme_path.write_text(content)

print("✅ README.md updated successfully!")
print("\nUpdated sections:")
print("  1. Baseline accuracy insight")
print("  2. Model size impact")
print("  3. Noise robustness")
print("  4. Figure 2 caption")
