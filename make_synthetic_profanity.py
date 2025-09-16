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
import numpy as np
from TTS.api import TTS
import soundfile as sf

# ------------------------------
# Config
# ------------------------------
random.seed(42)  # reproducibility

# Load input texts either from NSC or default list
def load_nsc_texts(path):
    texts = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    texts.append(line)
        return texts
    except Exception as e:
        print(f"Warning: Could not load NSC corpus ({e}), using default texts")
        return None

nsc_path = "data/NSC/transcripts.txt"
input_texts = load_nsc_texts(nsc_path) or [
    "I am going to the hawker centre for chicken rice",
    "Tomorrow we have meeting at the bank", 
    "The MRT is very crowded during peak hours",
    "Can you help me check the weather forecast",
    "My favorite food is laksa from Old Airport Road",
]

# Load profanity list (shared with evaluation pipeline)
with open("profanity_list.txt") as f:
    profanity_words = [line.strip() for line in f if line.strip()]

output_dir = Path("data/synthetic_profanity")
os.makedirs(output_dir, exist_ok=True)

# GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load multilingual TTS model
tts_model = TTS(model_name="tts_models/en/ljspeech/fast_pitch", progress_bar=False).to(device)

# Speaker variations for more diverse audio
SPEAKER_SETTINGS = [
    {"speed": 1.0, "voice_scale": 1.0},  # normal
    {"speed": 1.2, "voice_scale": 0.9},  # faster, higher pitch
    {"speed": 0.9, "voice_scale": 1.1},  # slower, lower pitch
]

# ------------------------------
# Helpers
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
failed = 0

def validate_audio(wav, sr):
    """Check if audio output is valid"""
    if not isinstance(wav, (np.ndarray, list)):
        return False
    if len(wav) == 0:
        return False
    if isinstance(wav, np.ndarray) and np.isnan(wav).any():
        return False
    return True

for i, clean_text in enumerate(input_texts):
    try:
        # Create multiple variations
        for variant, settings in enumerate(SPEAKER_SETTINGS):
            augmented_text = inject_profanity(clean_text, prob=0.8)  # 80% chance
            out_path = output_dir / f"utt_{i}_v{variant}.wav"

            # Synthesize speech with speaker variation
            wav = tts_model.tts(
                augmented_text,
                speed=settings["speed"],
                speaker_wav=None,
                language=None
            )
            
            # Validate audio output
            if not validate_audio(wav, tts_model.synthesizer.output_sample_rate):
                print(f"⚠️ Invalid audio generated for text: {augmented_text}")
                failed += 1
                continue

            # Apply voice scale variation
            wav = np.array(wav) * settings["voice_scale"]
            
            # Save audio
            sf.write(out_path, wav, tts_model.synthesizer.output_sample_rate)

            # Save metadata (relative path for HF compatibility)
            metadata.append({
                "audio": str(out_path.relative_to(output_dir)),
                "text": augmented_text,
                "clean_text": clean_text,
                "has_profanity": augmented_text != clean_text,
                "speed": settings["speed"],
                "voice_scale": settings["voice_scale"]
            })
    except Exception as e:
        print(f"⚠️ Failed to process text: {clean_text} ({e})")
        failed += 1

# ------------------------------
# Save JSONL
# ------------------------------
with open(output_dir / "metadata.jsonl", "w") as f:
    for row in metadata:
        f.write(json.dumps(row) + "\n")

print(f"✅ Generated {len(metadata)} synthetic samples with profanity at {output_dir}")
print(f"Failed generations: {failed}")

# Save dataset card for HuggingFace
with open(output_dir / "README.md", "w") as f:
    f.write(f"""# Synthetic Profanity Speech Dataset

Synthetic speech dataset for evaluating ASR safety and profanity handling.

- {len(input_texts)} base utterances
- {len(metadata)} total samples with speaker variations
- {len(SPEAKER_SETTINGS)} speaker variations per utterance
- {failed} failed generations
- Uses {tts_model.model_name} for TTS
""")