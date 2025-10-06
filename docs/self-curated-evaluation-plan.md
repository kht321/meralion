# Self-Curated Robustness Dataset Evaluation Plan

## Overview

This document outlines the plan to evaluate ASR models on a self-curated robustness dataset containing conversational audio samples. The dataset was prepared using the MERaLiON-2-3B model as a baseline transcription tool, with manual corrections to be applied.

## Dataset Details

### Location
- **Archive**: `data/robustness-20251005T175554Z-1-001.zip` (extracted to `data/robustness/`)
- **Audio files**: `data/robustness/audio/` (7 MP3 files: test1.mp3 through test7.mp3)
- **Metadata**: `data/robustness/metadata.xlsx`
- **Helper notebook**: `data/Self-Curated Transcription Helper.ipynb`

### Audio Characteristics
- **Format**: MP3
- **Count**: 7 audio samples
- **Total size**: ~9.5 MB compressed
- **Expected content**: Conversational speech, likely Singaporean/Southeast Asian accents
- **Sample durations**: Vary from ~500KB to ~3.6MB per file

### Transcription Baseline
The helper notebook shows that MERaLiON-2-3B was used to generate initial transcriptions with:
- **Model**: MERaLiON/MERaLiON-2-3B
- **Sample rate**: 16000 Hz
- **Prompt template**: "Instruction: Please transcribe this speech. \nFollow the text instruction based on the following audio: <SpeechHere>"
- **Output format**: Includes speaker tags (e.g., `<Speaker1>`, `<Speaker2>`)

## Evaluation Pipeline Integration

### Step 1: Extract and Prepare Ground Truth Transcripts

1. **Unzip dataset** (✓ Complete)
   ```bash
   cd data
   unzip robustness-20251005T175554Z-1-001.zip
   ```

2. **Extract metadata from Excel**
   - Convert `metadata.xlsx` to a readable format (JSON/CSV)
   - Extract columns:
     - `audio_file`: Filename (test1.mp3, test2.mp3, etc.)
     - `text`: Corrected ground truth transcript
     - Additional metadata fields (speaker info, accent type, etc.)

3. **Manual transcript correction workflow**
   - Use the helper notebook to generate MERaLiON-2-3B baseline transcripts
   - Listen to each audio file
   - Correct transcripts for:
     - Speech recognition errors
     - Speaker diarization accuracy
     - Singaporean English/Singlish expressions
     - Colloquialisms and code-switching

### Step 2: Create Manifest File

Create a JSONL manifest at `data/manifests/robustness_self_curated.jsonl` following the NSC format:

```python
# asr_eval/datasets/robustness_manifest.py

def build_robustness_manifest(root: Path, metadata_path: Path) -> List[dict]:
    """Build manifest for self-curated robustness dataset."""
    import pandas as pd

    # Read metadata (requires openpyxl)
    metadata = pd.read_excel(metadata_path)

    rows = []
    for idx, row in metadata.iterrows():
        audio_file = row['audio_file']  # e.g., 'test1.mp3'
        audio_path = root / 'audio' / audio_file

        if not audio_path.exists():
            continue

        rows.append({
            'audio': {'path': str(audio_path.relative_to(root.parent))},
            'audio_path': str(audio_path.relative_to(root.parent)),
            'utt_id': audio_path.stem,
            'text': row['text'].strip(),  # Ground truth transcript
            'speaker': row.get('speaker', 'unknown'),
            'accent': row.get('accent', 'sgp'),  # Singapore English
            'domain': 'conversational',
            'part': 'robustness_self_curated',
        })

    return rows
```

### Step 3: Update Configuration

Create evaluation config at `configs/robustness_self_curated.yaml`:

```yaml
# Self-curated robustness evaluation
dataset_manifest: "data/manifests/robustness_self_curated.jsonl"
dataset_audio_dir: "data/robustness"
text_field: "text"
results_dir: "results/robustness_self_curated"
seeds: [13, 17, 23]
models:
  - meralion-2-10b
  - meralion-2-3b
  - whisper-small

corruptions:
  none: [0]
  noise_snr_db: [30, 20, 10]
  speed: [0.9, 0.8, 1.1]
  pitch_semitones: [-2, 2, 4]
  reverb_decay: [0.2, 0.5, 0.8]
  clipping_ratio: [0.98, 0.9, 0.8]

bootstrap:
  n_samples: 1000
  alpha: 0.05
```

### Step 4: Run Evaluation

```bash
# 1. Generate manifest
python -m asr_eval.datasets.robustness_manifest \
    --root data/robustness \
    --metadata data/robustness/metadata.xlsx \
    --output data/manifests/robustness_self_curated.jsonl

# 2. Run robustness evaluation
python -m asr_eval.eval.run_robustness \
    --config configs/robustness_self_curated.yaml \
    --emit_jsonl

# Or use Makefile target
make robust-self-curated
```

### Step 5: Analysis and Comparison

Compare results with NSC Part 1 evaluation:
- **Dataset characteristics**:
  - NSC: Scripted speech, local accents, controlled recording
  - Self-curated: Conversational speech, natural dialogue, potentially noisy
- **Expected differences**:
  - Self-curated may show higher WER due to conversational nature
  - Speaker diarization challenges (multiple speakers)
  - Code-switching and Singlish expressions may increase error rates

## Implementation Checklist

- [ ] **Data Preparation**
  - [x] Unzip dataset archive
  - [ ] Install openpyxl dependency: `pip install openpyxl`
  - [ ] Extract and validate metadata from Excel
  - [ ] Listen to audio files and verify/correct transcripts
  - [ ] Document transcript correction guidelines

- [ ] **Code Implementation**
  - [ ] Create `asr_eval/datasets/robustness_manifest.py`
  - [ ] Add manifest builder function
  - [ ] Add CLI entrypoint (`if __name__ == "__main__"`)
  - [ ] Write unit test for manifest builder

- [ ] **Configuration**
  - [ ] Create `configs/robustness_self_curated.yaml`
  - [ ] Update Makefile with `robust-self-curated` target

- [ ] **Execution**
  - [ ] Generate manifest JSONL
  - [ ] Run robustness evaluation
  - [ ] Verify results in `results/robustness_self_curated/`

- [ ] **Documentation**
  - [ ] Update main README with self-curated dataset details
  - [ ] Document findings and model comparisons
  - [ ] Add to evaluation suite documentation

## Expected Outputs

1. **Manifest**: `data/manifests/robustness_self_curated.jsonl`
   - 7 utterances (one per audio file)
   - Ground truth transcripts with speaker tags

2. **Results**:
   - `results/robustness_self_curated/per_seed.csv`: Per-seed metrics
   - `results/robustness_self_curated/summary.csv`: Aggregated metrics with confidence intervals
   - `results/robustness_self_curated/details.jsonl`: Per-utterance transcriptions

3. **Metrics**:
   - WER/CER on clean audio
   - Robustness under corruption (noise, speed, pitch, reverb, clipping)
   - Comparison across MERaLiON-2-10B, MERaLiON-2-3B, and Whisper-small

## Notes and Considerations

### Transcript Quality
- The helper notebook shows MERaLiON-2-3B generates transcripts with speaker tags
- Manual correction is critical for accurate evaluation
- Pay attention to:
  - Singaporean English expressions (e.g., "Blk. Pasar")
  - Colloquialisms and hesitations (e.g., "(hm)", "(um)")
  - Speaker turn boundaries

### Dataset Size
- Only 7 samples - much smaller than NSC Part 1 (682 utterances)
- Statistical significance may be limited
- Consider as a qualitative supplement to NSC evaluation
- Bootstrap confidence intervals may be wide

### Evaluation Focus
- **Conversational speech**: Natural dialogue patterns
- **Speaker diarization**: Multi-speaker handling
- **Singaporean English**: Local accent and expressions
- **Robustness**: Same corruption grid as NSC for consistency

## Integration with Existing Pipeline

The self-curated dataset follows the same evaluation framework as NSC:
1. **Manifest format**: Compatible JSONL structure
2. **Corruption pipeline**: Reuses existing `asr_eval/corruption/` modules
3. **Metrics**: Same WER/CER calculation and bootstrap CI
4. **Config format**: Consistent YAML structure

This ensures direct comparability with NSC Part 1 results while providing insights into conversational speech performance.
