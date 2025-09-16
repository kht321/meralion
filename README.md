# MERALION — Synthetic Profanity & Evaluation MVP

Generate, augment, and evaluate **speech datasets with profanity and sensitive content**.  
The MVP ships:
- a profanity lexicon (`make_synthetic_profanity.py`),
- a dataset builder + splitter (`mvp.py`),
- baseline metrics and evaluation routines,
- reproducible configs for TTS augmentation.

**Objective:** reproducibly measure and *stress-test* speech/LLM pipelines against profanity, offensive language, and edge-case content.  

---

## What this does

1. **Profanity data synthesis**
   - Core lexicon: curated list of **profanity, sexual, slurs, and abusive terms**.
   - TTS augmentation: generates **synthetic speech** with Coqui-TTS.
   - Output: text + audio dataset (`.jsonl` with `.wav` paths).

2. **Dataset splitting**
   - `split_dataset()` — generic train/dev/test.
   - `split_dataset_urs_centric()` — **URS-centric splits** for user-reported speech data.
   - Outputs structured JSON with metadata for reproducibility.

3. **Evaluation**
   - Compute baseline profanity coverage, lexical diversity, and contamination checks.
   - Designed for **speech → transcript → LLM evaluation loops**.

---

## Results snapshot (current MVP)

Latest run (**data/synthetic_profanity/metadata.jsonl**, 15 clips, 22.05 kHz → 16 kHz resample):

- **Whisper-small (openai/whisper-small)**  
  - Precision ≈ **1.00** (8 TP / 0 FP) — never hallucinated profanity.  
  - Recall ≈ **0.62** (8 TP / 5 FN) — caught eight of thirteen injected profanities, missed five.  
  - F1 ≈ **0.76** with two clean utterances correctly marked safe.  
- **MERaLiON-2-10B**  
  - Currently returns **no predictions** (all metrics 0). Processor still emits `NoneType.ndim` errors on the resampled audio, so debugging is ongoing.

Interpretation: Whisper is already reliable at avoiding false positives but still misses some injected terms—fine-tuning or prompt conditioning should focus on recall. MERaLiON needs additional audio preprocessing/debugging before its numbers are meaningful.

---

## Simple summary (for non-technical readers)

- **What is this for?** To check how AI models handle **bad words** (swearing, sexual, hate).  
- **What did we build?**  
  - A **list of bad words** (text).  
  - A tool to **turn them into audio** (using TTS).  
  - A script to **split datasets** (train/test/dev).  
- **Does it work?** Yes. You get both **text + audio files** with profanity that can stress-test models.  
- **Why it matters?** AI speech/LLM systems often fail or behave strangely with profanity. This lets us measure and improve robustness.  
- **How to use it?**  
  ```bash
  python make_synthetic_profanity.py --out ./data/profanity.jsonl
  python mvp.py split --in ./data/profanity.jsonl --out ./splits/
  ```

---

## Repo layout

```
.
├─ mvp.py                       # dataset splitter + evaluator
├─ make_synthetic_profanity.py  # build text/audio profanity dataset
├─ requirements.txt             # core dependencies
├─ requirements-lock.txt        # frozen environment
├─ data/                        # datasets (JSONL + wav)
├─ splits/                      # structured splits
└─ results/                     # evaluation outputs
```

---

## Install

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip wheel setuptools
pip install -r requirements.txt
```

---

## Usage

### 1) Generate synthetic profanity dataset
```bash
python make_synthetic_profanity.py \
  --out ./data/profanity.jsonl \
  --tts voice_en \
  --n_samples 500
```
Output: `.jsonl` rows with text + `.wav` audio files.

### 2) Split into train/dev/test
```bash
python mvp.py split \
  --in ./data/profanity.jsonl \
  --out ./splits/ \
  --strategy flat
```

### 3) URS-centric splitting (user speech focus)
```bash
python mvp.py split-urs \
  --in ./data/profanity.jsonl \
  --out ./splits/urs/
```

### 4) Run evaluation
```bash
python mvp.py eval \
  --data ./splits/test.jsonl \
  --metrics coverage diversity balance \
  --out ./results/eval.json
```

---

## Outputs and formats

### Dataset row
```json
{
  "id": "profanity-001",
  "text": "fuck you",
  "audio": "./data/audio/profanity-001.wav",
  "category": "sexual",
  "metadata": {"length_sec": 1.2}
}
```

### Split file
```json
{
  "train": ["profanity-001", "profanity-002"],
  "dev": ["profanity-100"],
  "test": ["profanity-200"]
}
```

### Metrics file
```json
{
  "coverage": 0.98,
  "diversity": 0.74,
  "balance": {"train": 0.65, "dev": 0.17, "test": 0.18}
}
```

---

## Troubleshooting

- **Pylance “missing imports”** → select the `.venv` interpreter in VS Code.  
- **`torch` not found / MPS disabled** → reinstall Apple-Silicon wheels: `pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu`.  
- **Coqui-TTS errors** → ensure Python ≤3.11 (not 3.12).  
- **Audio silent** → check `soundfile` backend installed (`brew install libsndfile`).  

---

## License and attribution

This repository is part of the **AI Evaluation Group Project (MVP)**.  
Free for internal coursework use.  
For external use, please request permission.  
