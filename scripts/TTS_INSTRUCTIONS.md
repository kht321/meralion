
# TTS Audio Generation Instructions

## Option 1: Microsoft Azure TTS (Recommended for Singapore English)

Use the `en-SG-LunaNeural` or `en-SG-WayneNeural` voice for authentic Singapore accent.

```python
import azure.cognitiveservices.speech as speechsdk
from pathlib import Path

# Configure Azure Speech
speech_config = speechsdk.SpeechConfig(
    subscription="YOUR_KEY",
    region="YOUR_REGION"
)
speech_config.speech_synthesis_voice_name = "en-SG-LunaNeural"

# Set audio config
audio_config = speechsdk.audio.AudioOutputConfig(
    filename="output.wav"
)

# Synthesize
synthesizer = speechsdk.SpeechSynthesizer(
    speech_config=speech_config,
    audio_config=audio_config
)
result = synthesizer.speak_text_async("Your text here").get()
```

## Option 2: Google Cloud TTS

Use `en-SG` language code with `en-SG-Neural2-A` voice.

## Option 3: Human Recording (Most Authentic)

For best results, recruit native Singlish speakers to record:
1. Use a quiet room with minimal background noise
2. Record in WAV format, mono channel, 16kHz sample rate
3. Speak naturally with authentic Singlish intonation
4. Save files with naming convention: {category}/realistic_{idx:02d}.wav

## Post-Processing

After generating audio files:
1. Ensure all files are mono, 16kHz, WAV format
2. Place in: data/guardrails/audio/{category}/realistic_{idx:02d}.wav
3. Verify alignment with transcripts_realistic.json
4. Run validation: python scripts/validate_guardrail_dataset.py
