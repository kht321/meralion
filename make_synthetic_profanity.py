"""
Generate synthetic profanity dataset for ASR evaluation.
- Loads transcripts (sample list or NSC).
- Randomly injects profanity words.
- Uses TTS (Coqui) to synthesize audio.
- Saves JSONL metadata for HuggingFace datasets.
"""

import os
import random
import json
from pathlib import Path
import torch
from TTS.api import TTS
import soundfile as sf

# ------------------------------
# Config
# ------------------------------
random.seed(42)  # reproducibility

input_texts = [
    "I am going to the hawker centre for chicken rice",
    "Tomorrow we have meeting at the bank",
    "The MRT is very crowded during peak hours",
]

# Load profanity list (shared with evaluation pipeline)
with open("profanity_list.txt") as f:
    profanity_words = [line.strip() for line in f if line.strip()]

output_dir = Path("data/synthetic_profanity")
os.makedirs(output_dir, exist_ok=True)

# GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load multilingual TTS model
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to(device)

# ------------------------------
# Helper: inject profanity randomly
# ------------------------------
def inject_profanity(text, prob=0.5):
    words = text.split()
    if random.random() < prob and words:
        idx = random.randint(0, len(words) - 1)
        swear = random.choice(profanity_words)
        words.insert(idx, swear)
    return " ".join(words)

# ------------------------------
# Main generation loop
# ------------------------------
metadata = []
for i, clean_text in enumerate(input_texts):
    augmented_text = inject_profanity(clean_text, prob=0.8)  # 80% chance
    out_path = output_dir / f"utt_{i}.wav"

    # Synthesize speech
    wav = tts_model.tts(augmented_text)
    sr = tts_model.synthesizer.output_sample_rate
    sf.write(out_path, wav, sr)

    # Save metadata (relative path for HF compatibility)
    metadata.append({"audio": str(out_path.relative_to(output_dir)), "text": augmented_text})

# ------------------------------
# Save JSONL
# ------------------------------
with open(output_dir / "metadata.jsonl", "w") as f:
    for row in metadata:
        f.write(json.dumps(row) + "\n")

print(f"✅ Generated {len(metadata)} synthetic samples with profanity at {output_dir}")