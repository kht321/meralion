#!/usr/bin/env python3
"""Update final_report.py with corrected metrics."""

from pathlib import Path

report_path = Path("scripts/final_report.py")
content = report_path.read_text()

# Update 1: Abstract robustness values
content = content.replace(
    "exceptional robustness (+0.3-0.5pp average degradation under noise, speed, pitch, reverb, and clipping)",
    "exceptional robustness (+0.5-0.6pp average degradation under noise, speed, pitch, reverb, and clipping) and superior baseline accuracy (13.6% WER vs Whisper's 17.9%)"
)

# Update 2: Table 1 - NSC Robustness results
old_table1 = '''        ["MERaLiON-2-10B", "26.1%", "22.1%", "+0.3pp", "+6.2pp", "Noise 10dB (32.3%)"],
        ["MERaLiON-2-3B", "29.0%", "29.3%", "+0.5pp", "+5.1pp", "Noise 10dB (34.1%)"],'''

new_table1 = '''        ["MERaLiON-2-10B", "13.6%", "3.3%", "+0.6pp", "+5.8pp", "Noise 10dB (19.4%)"],
        ["MERaLiON-2-3B", "13.1%", "3.1%", "+0.5pp", "+5.1pp", "Noise 10dB (18.2%)"],'''

content = content.replace(old_table1, new_table1)

# Update 3: Table 2 - Conversational results
old_table2 = '''        ["MERaLiON-2-10B", "39.8%", "+0.8pp", "+5.3pp (Speed 0.8x)", "+13.7pp"],
        ["MERaLiON-2-3B", "23.8%", "+3.5pp", "+12.3pp (Noise 10dB)", "-5.2pp"],
        ["Whisper-small", "38.0%", "+3.6pp", "+23.8pp (Reverb 0.8)", "+20.1pp"]'''

new_table2 = '''        ["MERaLiON-2-10B", "66.5%", "-8.5pp", "+0.4pp (Clipping 0.8)", "+52.9pp"],
        ["MERaLiON-2-3B", "38.8%", "+2.8pp", "+28.2pp (Noise 10dB)", "+25.7pp"],
        ["Whisper-small", "52.2%", "+6.4pp", "+38.5pp (Reverb 0.8)", "+34.3pp"]'''

content = content.replace(old_table2, new_table2)

# Update 4: Conversational discussion paragraph
old_conv_para = '''        "The conversational dataset revealed a striking model size reversal: MERaLiON-2-3B achieved 23.8% WER, dramatically "
        "outperforming the 10B variant (39.8%, a 16pp gap). This reverses the NSC pattern where larger models excelled, "'''

new_conv_para = '''        "The conversational dataset revealed a striking model size reversal: MERaLiON-2-3B achieved 38.8% WER, dramatically "
        "outperforming the 10B variant (66.5%, a 27.7pp gap). This reverses the NSC pattern where models were nearly tied (0.5pp difference), "'''

content = content.replace(old_conv_para, new_conv_para)

# Update 5: Domain shift numbers
content = content.replace(
    'from 17.9% (NSC) to 38.0% (conversational)',
    'from 17.9% (NSC) to 52.2% (conversational)'
)

# Update 6: Conclusion robustness values
old_conclusion = '''        "This comprehensive evaluation demonstrates MERaLiON's exceptional acoustic robustness (+0.3-0.5pp average degradation) "'''

new_conclusion = '''        "This comprehensive evaluation demonstrates MERaLiON's exceptional acoustic robustness (+0.5-0.6pp average degradation) AND superior baseline accuracy (13.6% vs 17.9% WER), decisively outperforming Whisper. "'''

content = content.replace(old_conclusion, new_conclusion)

# Update 7: Conclusion conversational values
content = content.replace(
    "conversational superiority (23.8% vs. 39.8% WER)",
    "conversational superiority (38.8% vs. 66.5% WER for 3B vs 10B)"
)

# Write updated content
report_path.write_text(content)

print("✅ final_report.py updated successfully!")
print("\nUpdated sections:")
print("  1. Abstract - robustness and baseline accuracy")
print("  2. Table 1 - NSC robustness results")
print("  3. Table 2 - Conversational results")
print("  4. Conversational discussion paragraph")
print("  5. Domain shift numbers")
print("  6. Conclusion - robustness and baseline")
print("  7. Conclusion - conversational values")
