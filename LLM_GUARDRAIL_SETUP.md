# Layer 3 LLM-Based Safety Classifier Setup Guide

## Overview

The MERaLiON evaluation framework includes an **optional Layer 3 LLM-based safety classifier** that uses Groq's Llama-3.1-8B model for semantic content analysis and masking.

### Three-Layer Defense Architecture

1. **Layer 1: Logit-level Token Masking** - Prevents harmful tokens during generation
2. **Layer 2: Regex-based Post-processing** - Keyword matching with pattern matching
3. **Layer 3: LLM-based Safety Classifier** *(optional)* - Semantic understanding with context-aware classification

## Prerequisites

- Groq API account (free tier available)
- `groq` Python package (already in requirements.txt)

## Setup Instructions

### 1. Get Your Groq API Key

1. Visit [https://console.groq.com/keys](https://console.groq.com/keys)
2. Sign up or log in to your Groq account
3. Create a new API key
4. Copy the key (it will look like: `gsk_...`)

### 2. Configure Environment Variable

**Option A: Using .env file (Recommended)**

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API key
# Replace 'your_groq_api_key_here' with your actual key
GROQ_API_KEY=gsk_your_actual_key_here
```

**Option B: Export environment variable**

```bash
# For current session only
export GROQ_API_KEY=gsk_your_actual_key_here

# Or add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
echo 'export GROQ_API_KEY=gsk_your_actual_key_here' >> ~/.zshrc
source ~/.zshrc
```

**Option C: Pass directly via command line**

```bash
python -m asr_eval.whitebox.run_guardrail_eval \
  --model meralion-2-10b \
  --use-safety-classifier \
  --groq-api-key gsk_your_actual_key_here
```

### 3. Verify Installation

```bash
# Test that groq package is installed
python -c "import groq; print('✓ Groq package installed')"

# Test API key is configured
python -c "import os; print('✓ API key configured' if os.getenv('GROQ_API_KEY') else '✗ API key not found')"
```

## Usage

### Running Guardrail Evaluation with LLM Classifier

```bash
# Basic usage (reads GROQ_API_KEY from environment)
python -m asr_eval.whitebox.run_guardrail_eval \
  --model meralion-2-10b \
  --use-safety-classifier

# With all layers enabled
python -m asr_eval.whitebox.run_guardrail_eval \
  --model meralion-2-10b \
  --use-safety-classifier \
  --capture-decoder-trace

# Custom output directory
python -m asr_eval.whitebox.run_guardrail_eval \
  --model meralion-2-10b \
  --use-safety-classifier \
  --output-dir results/guardrails_llm
```

### Programmatic Usage

```python
from asr_eval.models.meralion import MERaLiON
from asr_eval.device import Device
import os

# Initialize with API key
device = Device()
api_key = os.getenv("GROQ_API_KEY")
model = MERaLiON("MERaLiON/MERaLiON-2-10B", device, groq_api_key=api_key)

# Enable the safety classifier
model.enable_safety_classifier()

# Transcribe with LLM-based safety processing
metadata = model.transcribe(audio, sample_rate, return_metadata=True)

print(f"Raw transcript: {metadata['raw']}")
print(f"Safety processed: {metadata['safety_processed']}")
print(f"Final text: {metadata['final_text']}")
```

## How It Works

The Layer 3 classifier performs three operations:

1. **Classification**: Categorizes text into `benign`, `hate_speech`, or `violence`
2. **Profanity Masking**: Replaces profanity with `[CENSOREDTEXT]`
3. **PII Masking**: Replaces personal information with `[REDACTED]`

### Processing Flow

```
Raw ASR Output
    ↓
Layer 3: LLM Classifier (if enabled)
    ├─ Classify: benign/hate_speech/violence
    ├─ If benign: mask profanity + PII
    └─ If hate/violence: block with category message
    ↓
Layer 2: Regex Guardrail (if not handled by Layer 3)
    └─ Keyword pattern matching
    ↓
Final Output
```

## Security Notes

- ✅ `.env` files are gitignored by default
- ✅ Never commit API keys to git
- ✅ Use environment variables in production
- ✅ `.env.example` is included for documentation only

## Costs

- Groq offers generous free tier limits
- Llama-3.1-8B-instant is optimized for low latency
- Typical cost: ~$0.001 per 1000 evaluations (approximate)

## Troubleshooting

### "groq package is required"

```bash
pip install groq
# or
pip install -r requirements.txt
```

### "Groq API key required"

Check that your environment variable is set:

```bash
echo $GROQ_API_KEY
```

If empty, follow setup instructions above.

### "Failed to initialize SmartSafetyTranscriber"

1. Verify API key is valid
2. Check internet connection
3. Verify Groq service status at [status.groq.com](https://status.groq.com)

## Comparison: With vs. Without Layer 3

| Metric | Layer 1+2 Only | Layer 1+2+3 (with LLM) |
|--------|----------------|------------------------|
| Blocking Rate | 22.5% | TBD (requires re-run) |
| Latency Overhead | +39ms | TBD (LLM adds ~200-500ms) |
| False Positives | 0% | TBD |
| Context Understanding | ❌ No | ✅ Yes |
| PII Detection | ❌ Limited | ✅ Advanced |

## Next Steps

After setup, you can:

1. **Run evaluation** with `--use-safety-classifier` flag
2. **Compare results** between Layer 1+2 vs Layer 1+2+3
3. **Analyze trade-offs** between accuracy and latency
4. **Update documentation** with empirical results

## References

- Groq Documentation: [https://console.groq.com/docs](https://console.groq.com/docs)
- Llama 3.1 Model: [https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
