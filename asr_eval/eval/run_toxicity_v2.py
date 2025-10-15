"""
Toxicity evaluation for ASR models - Config-based multi-model version
Supports MERaLiON-2-10B, MERaLiON-2-3B, and Whisper-small
"""
import os, sys, argparse
from pathlib import Path
import torch
import librosa
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from sklearn.metrics import classification_report
from detoxify import Detoxify
import jiwer
import re
import json
import yaml

try:
    from asr_eval.metrics.wer_cer import corpus_scores
    from asr_eval.eval.run_robustness import build_model
    from asr_eval.device import Device
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from asr_eval.metrics.wer_cer import corpus_scores
    from asr_eval.eval.run_robustness import build_model
    from asr_eval.device import Device

def read_csv_robust(path):
    """Read CSV with multiple encoding attempts."""
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return pd.read_csv(f)

def find_audio_file(root: str, dataset: str, fname: str) -> str:
    """Find audio file with various extensions."""
    exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    candidates = []
    for ext in exts:
        candidates.append(os.path.join(root, f"{fname}{ext}"))
        candidates.append(os.path.join(root, dataset, f"{fname}{ext}"))

    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Audio not found for {fname} under {root}")

def transcribe_with_model(model, audio_path):
    """Transcribe audio using any ASR model (MERaLiON or Whisper)."""
    try:
        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        transcript = model.transcribe(waveform, sr)
        return transcript
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        return ""

def classify_toxicity_multiple(texts):
    """Run multiple toxicity classifiers on texts."""
    results = {}

    # 1. Toxic-BERT
    print("Running Toxic-BERT classification...")
    toxic_bert = pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        top_k=6,
        truncation=True,
        max_length=256,
        padding=True,
        batch_size=32,
        device=0 if torch.cuda.is_available() else -1,
    )
    toxic_bert_preds = toxic_bert(texts)
    results['bert_tox'] = [
        int(any(label["score"] >= 0.5 for label in sample_scores))
        for sample_scores in tqdm(toxic_bert_preds, desc="Processing Toxic-BERT")
    ]

    # 2. Detoxify
    print("Running Detoxify classification...")
    detoxify_model = Detoxify('original')
    detoxify_preds = []
    for text in tqdm(texts, desc="Running Detoxify"):
        try:
            scores = detoxify_model.predict(text if isinstance(text, str) else "")
            is_toxic = int(max(scores.values()) >= 0.5)
            detoxify_preds.append(is_toxic)
        except:
            detoxify_preds.append(0)
    results['detoxy_tox'] = detoxify_preds

    return results

def main():
    parser = argparse.ArgumentParser(description="Run toxicity evaluation on ASR models")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg['model_name']
    model_id = cfg['model_id']
    audio_root = cfg['audio_root']
    meta_path = cfg['meta_path']
    result_dir = cfg['result_dir']
    dataset_filter = cfg.get('dataset_filter', 'Common Voice')
    max_samples = cfg.get('max_samples')

    os.makedirs(result_dir, exist_ok=True)

    # Load and filter metadata
    print(f"Loading metadata from: {meta_path}")
    df = read_csv_robust(meta_path)
    df = df[df["Dataset"] == dataset_filter].reset_index(drop=True)

    if max_samples:
        df = df.head(max_samples)

    print(f"Loaded {len(df)} {dataset_filter} samples")

    # Load model
    device = Device()
    print(f"Loading {model_name} from {model_id} on {device}")
    model = build_model(model_name, device)

    # Process each audio file
    print("Processing audio files...")
    transcripts = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Transcribing"):
        try:
            file_path = find_audio_file(audio_root, row["Dataset"], row["FileName"])
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            transcripts.append("")
            continue

        transcript = transcribe_with_model(model, file_path)
        transcripts.append(transcript)

    df["pred_text"] = transcripts

    # Run toxicity classifiers on transcripts
    print("\\nRunning toxicity classification on transcripts...")
    toxicity_results = classify_toxicity_multiple(df["pred_text"].tolist())
    for col, values in toxicity_results.items():
        df[col] = values

    # Compute WER/CER metrics
    print("Computing WER/CER metrics...")
    valid_pairs = [(ref, hyp) for ref, hyp in zip(df["text"], df["pred_text"])
                   if isinstance(ref, str) and isinstance(hyp, str) and ref and hyp]

    if valid_pairs:
        refs, hyps = zip(*valid_pairs)
        corpus_metrics = corpus_scores(list(refs), list(hyps))
        df["wer"] = [
            jiwer.wer(ref if isinstance(ref, str) else "",
                      hyp if isinstance(hyp, str) else "")
            for ref, hyp in zip(df["text"], df["pred_text"])
        ]
        df["cer"] = [
            jiwer.cer(ref if isinstance(ref, str) else "",
                      hyp if isinstance(hyp, str) else "")
            for ref, hyp in zip(df["text"], df["pred_text"])
        ]
        wer_summary = (
            f"Corpus WER: {corpus_metrics.wer:.4f}\\n"
            f"Corpus CER: {corpus_metrics.cer:.4f}\\n"
            f"Valid utterances: {len(valid_pairs)}/{len(df)}\\n"
        )
    else:
        df["wer"] = [float('inf')] * len(df)
        df["cer"] = [float('inf')] * len(df)
        wer_summary = "No valid reference-hypothesis pairs found\\n"

    # Generate classification reports
    print("Generating classification reports...")
    gold_labels = df["label2a"]
    reports = []
    for classifier in ["bert_tox", "detoxy_tox"]:
        if classifier in df.columns and df[classifier].notna().any():
            report = classification_report(
                gold_labels, df[classifier],
                target_names=['Non-toxic', 'Toxic'],
                output_dict=False
            )
            reports.append(f"\\n{classifier.upper()} Classification Report:\\n{report}")

    # Save results
    print(f"Saving results to: {result_dir}")
    with open(os.path.join(result_dir, f"{model_name}_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"TOXICITY EVALUATION RESULTS - {model_name.upper()}\\n")
        f.write("=" * 60 + "\\n")
        f.write(wer_summary)
        f.write("\\n")
        for report in reports:
            f.write(report)
            f.write("\\n")

    # Save CSV
    output_columns = ["ref_text", "pred_text", "gold_tox", "bert_tox", "detoxy_tox", "wer", "cer"]
    output_df = df.rename(columns={"text": "ref_text", "label2a": "gold_tox"})

    # Only keep columns that exist
    existing_cols = [c for c in output_columns if c in output_df.columns]
    output_df[existing_cols].to_csv(
        os.path.join(result_dir, f"{model_name}_results.csv"),
        index=False
    )

    # Print summary
    print("\\nEVALUATION SUMMARY:")
    print("=" * 60)
    print(wer_summary)
    for report in reports:
        print(report)
    print(f"\\nResults saved to:")
    print(f"- {result_dir}/{model_name}_summary.txt")
    print(f"- {result_dir}/{model_name}_results.csv")
    print("Evaluation completed!")

    # Cleanup
    if hasattr(model, 'close'):
        model.close()

if __name__ == "__main__":
    main()
