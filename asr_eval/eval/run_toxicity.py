"""
Toxicity evaluation for ASR models (config-based, multi-model)
Supports MERaLiON-2-10B, MERaLiON-2-3B, and Whisper-small.
"""

import os
import sys
import argparse
import re
import string
import logging
from pathlib import Path

import torch
import librosa
import pandas as pd
import yaml
import jiwer
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

try:
    from asr_eval.metrics.wer_cer import corpus_scores
    from asr_eval.eval.run_robustness import build_model
    from asr_eval.device import Device
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from asr_eval.metrics.wer_cer import corpus_scores
    from asr_eval.eval.run_robustness import build_model
    from asr_eval.device import Device


# Setup logging (root logger)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)



def read_csv_robust(path):
    """Read CSV file with multiple encoding fallbacks."""
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return pd.read_csv(f)


def find_audio_file(root: str, dataset: str, fname: str) -> str:
    """Find audio file by trying common extensions and dataset subfolders."""
    exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    candidates = [os.path.join(root, f"{fname}{ext}") for ext in exts]
    candidates += [os.path.join(root, dataset, f"{fname}{ext}") for ext in exts]
    search_dirs = [root, os.path.join(root, dataset)]
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        for file in os.listdir(search_dir):
            base, ext = os.path.splitext(file)
            if base == fname and ext.lower() in exts:
                return os.path.join(search_dir, file)
    raise FileNotFoundError(f"Audio not found for {fname} under {root}")


def clean_response(text):
    """Remove 'model' prefix and speaker tags, but do NOT remove square brackets. Normalize whitespace/case."""
    text = re.sub(r"^(model\s*<[^>]+>:|model\s+speaker\s*\d+\s*:?|user\s+model\s*:?)", "", 
                  text, flags=re.IGNORECASE)
    text = text.strip()
    # Remove all punctuation (including square brackets)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text


def extract_toxicity_decision(response):
    """Extract binary decision (0/1) from toxicity classification response."""
    s = response.strip().lower()
    
    # Check for common toxic indicators
    if s in {"1", "true", "toxic", "yes"}:
        return 1
    
    # Try JSON parsing
    import json as _json
    json_start = s.find('{')
    json_end = s.rfind('}')
    if json_start != -1 and json_end != -1 and json_end > json_start:
        try:
            data = _json.loads(s[json_start:json_end + 1])
            lbl = data.get("label")
            if isinstance(lbl, bool):
                return 1 if lbl else 0
            if isinstance(lbl, (int, float)):
                return 1 if int(lbl) == 1 else 0
            if isinstance(lbl, str):
                return 1 if lbl.strip().lower() in {"1", "true", "toxic", "yes"} else 0
        except Exception:
            pass
    
    return 0


def is_meralion_model(model_id):
    """Check if model is MERaLiON."""
    return isinstance(model_id, str) and 'meralion' in model_id.lower()


def process_audio_samples(df, model, model_id, audio_root, no_toxic):
    """
    Process audio samples:
    - MERaLiON: Transcribe, then classify audio toxicity directly
    - Other models: Transcribe only
    """
    from asr_eval.models.meralion import NO_TOXIC_TRANSCRIBE_PROMPT
    
    is_meralion = is_meralion_model(model_id)
    use_no_toxic_prompt = bool(no_toxic)
    
    transcripts_clean = []
    blocked_counts = []
    tox_responses = []
    tox_labels = []

    logger.debug("Processing audio files...")

    blocked_files = []
    transcription_errors = []
    classification_errors = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing audio"):
        fname = row["FileName"]
        try:
            file_path = find_audio_file(audio_root, row["Dataset"], fname)
        except FileNotFoundError as e:
            logger.warning(f"Audio file not found: {e}")
            transcripts_clean.append("")                
            if is_meralion:
                tox_responses.append("")
                tox_labels.append(0)
                blocked_counts.append(0)
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing sample {idx + 1}/{len(df)}: {fname}")

        # Load audio
        waveform, sr = librosa.load(file_path, sr=16000, mono=True)
        
        logger.debug(f"Audio loaded: shape={waveform.shape}, sr={sr}")
        
        # Transcribe audio
        if is_meralion and use_no_toxic_prompt and hasattr(model, "set_prompt"):
            model.set_prompt(NO_TOXIC_TRANSCRIBE_PROMPT)
            logger.info("[TRANSCRIBE] Using NO_TOXIC_TRANSCRIBE_PROMPT")
        
        try:
            if is_meralion:
                raw_transcript = model.transcribe(waveform, sr, return_metadata=False, use_logits_masking=False)
                                
                cleaned_transcript = clean_response(raw_transcript)
                transcripts_clean.append(cleaned_transcript)
                logger.info(f"[TRANSCRIBE] Cleaned transcript: {cleaned_transcript}")
                                
                blocked_count = raw_transcript.count("censoredtext")
                blocked_counts.append(blocked_count)
                
                if blocked_count > 0:
                    blocked_files.append((fname, blocked_count))
                    logger.info(f"{fname} has {blocked_count} blocked token(s)")
            else:
                raw_transcript = model.transcribe(waveform, sr)
                cleaned_transcript = clean_response(raw_transcript)
                transcripts_clean.append(cleaned_transcript)
                logger.info(f"[TRANSCRIBE] Cleaned transcript: {cleaned_transcript}")

                blocked_counts.append("")

        except Exception as e:
            logger.error(f"Error transcribing {fname}: {e}")
            transcripts_clean.append("")
            transcription_errors.append((fname, str(e)))
        
        # For MERaLiON: classify audio toxicity
        if is_meralion:
            try:
                raw_tox_response = model.classify_toxicity(waveform, sr)
                
                tox_response = clean_response(raw_tox_response)
                tox_responses.append(tox_response)
                logger.info(f"[CLASSIFY AUDIO] Cleaned response: {tox_response}")
                
                tox_label = extract_toxicity_decision(tox_response)
                tox_labels.append(tox_label)
                logger.info(f"[CLASSIFY AUDIO] Extracted label: {tox_label}")

            except Exception as e:
                logger.error(f"Error classifying audio toxicity for {fname}: {e}")
                tox_responses.append("")
                tox_labels.append(None)
                classification_errors.append((fname, str(e)))
    
    df["pred"] = transcripts_clean
    df["blocked_count"] = blocked_counts
    if is_meralion:
        df["model_response"] = tox_responses
        df["model_tox"] = tox_labels
    
    # Summary statistics
    logger.info(f"\n{'='*60}")
    logger.info(f"\nProcessing complete:")
    logger.info(f"Total samples processed: {len(df)}")
    logger.info(f"Successful transcriptions: {len([t for t in transcripts_clean if t])}")
    
    if transcription_errors:
        logger.warning(f"  - Transcription errors: {len(transcription_errors)}")
        for fname, error in transcription_errors:
            logger.debug(f"    - {fname}: {error}")
    
    if is_meralion:
        logger.info(f"Successful classifications: {len([t for t in tox_labels if t is not None])}")
        if classification_errors:
            logger.warning(f"  - Classification errors: {len(classification_errors)}")
            for fname, error in classification_errors:
                logger.debug(f"    - {fname}: {error}")
        
        if blocked_files:
            logger.info(f"Samples with blocked tokens: {len(blocked_files)}")
            logger.debug("  - Files with blocked tokens:")
            for fname, count in blocked_files[:10]: 
                logger.debug(f"    - {fname}: {count} blocked token(s)")
            if len(blocked_files) > 10:
                logger.debug(f"    ... and {len(blocked_files) - 10} more")


def process_text_samples(df, model, model_id):
    """
    Process text samples:
    - MERaLiON: Classify reference text toxicity
    - Other models: Do nothing
    """
    is_meralion = is_meralion_model(model_id)

    if is_meralion:
        logger.info("MERaLiON detected: Classifying reference text toxicity")
        tox_responses = []
        tox_labels = []
        classification_errors = []

        for idx, ref_text in enumerate(tqdm(df["text"], desc="Classifying text")):
            logger.info(f"\n{'='*60}")
            logger.info(f"Classifying text sample {idx + 1}/{len(df)}")

            try:
                raw_tox_response = model.classify_toxicity(ref_text)
                
                tox_response = clean_response(raw_tox_response)
                tox_responses.append(tox_response)
                logger.info(f"[CLASSIFY TEXT] Cleaned response: {tox_response}")

                tox_label = extract_toxicity_decision(tox_response)
                tox_labels.append(tox_label)
                logger.info(f"[CLASSIFY TEXT] Extracted label: {tox_label}")
                
            except Exception as e:
                logger.error(f"Error classifying text toxicity (sample {idx + 1}): {e}")
                tox_responses.append("")
                tox_labels.append(None)
                classification_errors.append((idx + 1, str(e)))

        df["model_response"] = tox_responses
        df["model_tox"] = tox_labels

        # Summary statistics
        logger.info(f"\n{'='*60}")
        logger.info(f"\nText classification complete:")
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Successful classifications: {len([t for t in tox_labels if t is not None])}")
        if classification_errors:
            logger.warning(f"  - Classification errors: {len(classification_errors)}")
            for sample_num, error in classification_errors:
                logger.debug(f"    - Sample {sample_num}: {error}")
    else:
        logger.info("Non-MERaLiON model with text input: No processing needed")


def compute_wer_cer(df, input_type):
    """Compute WER/CER metrics if transcripts are available."""
    if "pred" not in df.columns:
        return "N/A - No transcripts generated\n"
    
    if input_type == "audio":
        logger.info("Computing WER/CER metrics...")

    refs_clean = df["ref_clean"].tolist()
    preds_clean = df["pred"].tolist()
    valid_pairs = [(ref, hyp) for ref, hyp in zip(refs_clean, preds_clean) if ref and hyp]
    
    if valid_pairs:
        refs, hyps = zip(*valid_pairs)
        corpus_metrics = corpus_scores(list(refs), list(hyps))
        df["wer"] = [jiwer.wer(ref, hyp) if ref and hyp else float('inf') 
                     for ref, hyp in zip(refs_clean, preds_clean)]
        df["cer"] = [jiwer.cer(ref, hyp) if ref and hyp else float('inf') 
                     for ref, hyp in zip(refs_clean, preds_clean)]
        return (
            f"Corpus WER: {corpus_metrics.wer:.4f}\n"
            f"Corpus CER: {corpus_metrics.cer:.4f}\n"
            f"Valid utterances: {len(valid_pairs)}/{len(df)}\n"
        )
    else:
        df["wer"] = [float('inf')] * len(df)
        df["cer"] = [float('inf')] * len(df)
        return "No valid reference-hypothesis pairs found\n"


def generate_reports(df, model_id):
    """Generate classification reports for toxicity predictions."""
    if df.get("model_tox") is None:
        return []
    
    gold_labels = df["label2a"]
    reports = []
    
    if is_meralion_model(model_id):
        logger.info("Generating classification reports...")
        report = classification_report(
            gold_labels, df["model_tox"],
            labels=[0, 1],
            target_names=['Non-toxic', 'Toxic'],
            output_dict=False
        )
        reports.append(f"\nMERaLiON Classification Report:\n{report}")
    
    return reports


def save_results(df, config_path, result_dir, model_id):
    """Save results to Excel with multiple sheets."""
    
    output_df = df.rename(columns={"text": "ref_text", "label2a": "gold_tox"})
    output_columns = [
        "ref_text", "ref_clean", "pred", "gold_tox",
        "model_tox", "model_response", "blocked_count", "wer", "cer",
    ]
    existing_cols = [c for c in output_columns if c in output_df.columns]
    
    filename_base = os.path.basename(config_path).split('.')[0]
    xlsx_path = os.path.join(result_dir, f"{filename_base}_results.xlsx")
    
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        output_df[existing_cols].to_excel(writer, sheet_name="Results", index=False)
        
        # Add confusion matrix and classification report sheets
        if "model_tox" in df.columns:
            gold_labels = df["label2a"]
            preds = df["model_tox"]

            if is_meralion_model(model_id) and preds.notna().any():
                cm = confusion_matrix(gold_labels, preds, labels=[0, 1])
                cr = classification_report(
                    gold_labels, preds, labels=[0, 1],
                    target_names=["Non-toxic", "Toxic"],
                    output_dict=True
                )
                
                cm_df = pd.DataFrame(
                    cm,
                    index=["Non-toxic", "Toxic"],
                    columns=["Pred Non-toxic", "Pred Toxic"]
                )
                cr_df = pd.DataFrame(cr).transpose()
                
                cm_df.to_excel(writer, sheet_name="model_tox_ConfMat")
                cr_df.to_excel(writer, sheet_name="model_tox_ClassRpt")
    
    return xlsx_path


def main():
    parser = argparse.ArgumentParser(description="Run toxicity evaluation on ASR models")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()

    # Setup file logging
    filename_base = os.path.basename(args.config).split('.')[0]
    result_dir = None
    log_path = None
    # Load config to get result_dir
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    result_dir = cfg['result_dir']
    os.makedirs(result_dir, exist_ok=True)
    log_path = os.path.join(result_dir, f"{filename_base}_log.txt")

    # Remove all handlers before adding file and console handler (root logger)
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(file_handler)
    # Add only one console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)
    root_logger.propagate = False
    global logger
    logger = logging.getLogger()

    model_name = cfg['model_name']
    model_id = cfg['model_id']
    audio_root = cfg['audio_root']
    meta_path = cfg['meta_path']
    dataset_filter = cfg.get('dataset_filter', 'Common Voice')
    max_samples = cfg.get('max_samples')
    input_type = cfg.get('input_type', 'audio')
    no_toxic = cfg.get('no_toxic', False)

    # Load and filter metadata
    logger.info(f"Loading metadata from: {meta_path}")
    df = read_csv_robust(meta_path)
    df = df[df["Dataset"] == dataset_filter].reset_index(drop=True)
    if max_samples:
        df = df.head(max_samples)
    logger.info(f"Loaded {len(df)} {dataset_filter} samples")

    # Add cleaned reference column
    df["ref_clean"] = [clean_response(ref) if isinstance(ref, str) else "" 
                       for ref in df["text"]]

    # Load ASR model
    device = Device()
    logger.info(f"Loading {model_name} from {model_id} on {device}")
    model = build_model(model_name, device)

    # Process samples based on input type and model
    logger.info(f"Input type: {input_type}")
    logger.info(f"MERaLiON model detected: {is_meralion_model(model_id)}")

    if input_type == "audio":
        process_audio_samples(df, model, model_id, audio_root, no_toxic)
        wer_summary = compute_wer_cer(df, input_type)
    elif input_type == "text":
        process_text_samples(df, model, model_id)
        wer_summary = "WER/CER: N/A - Text input only\n"
    else:
        raise ValueError(f"Unknown input_type: {input_type}. Must be 'audio' or 'text'.")

    # Clean output columns based on input type and model
    if input_type == "text":
        df = df.drop(columns=["blocked_count", "wer", "cer"], errors='ignore')
    if not is_meralion_model(model_id):
        df = df.drop(columns=["gold_tox", "model_tox", "model_response", "blocked_count"], errors='ignore')

    # Generate classification reports
    reports = generate_reports(df, model_id)

    # Save results
    save_results(df, args.config, result_dir, model_id)

    # Print summary
    print("\nEVALUATION SUMMARY:")
    print("=" * 60)
    print(wer_summary)
    for report in reports:
        print(report)

    logger.info(f"Saving results to: {result_dir}")
    print("\nEvaluation completed!")

    if hasattr(model, 'close'):
        model.close()


if __name__ == "__main__":
    main()