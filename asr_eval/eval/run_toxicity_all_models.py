import os, sys
from pathlib import Path
import torch
import librosa
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from sklearn.metrics import classification_report
from detoxify import Detoxify
import jiwer
import re
import json

try:
    from asr_eval.metrics.wer_cer import corpus_scores
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from asr_eval.metrics.wer_cer import corpus_scores

# Configuration
MODEL_NAME = "MERaLiON-2-3B"
AUDIO_ROOT = "data/detoxy/audio/test"
META_PATH = "data/detoxy/metadata/test.csv"
RESULT_DIR = "results/toxicity"
REPO_ID = "asr_eval/models_local/MERaLiON-2-3B"

os.makedirs(RESULT_DIR, exist_ok=True)

# MERaLiON prompt templates
PROMPT_TEMPLATE = "Instruction: {query} \nFollow the text instruction based on the following audio: <SpeechHere>"
TRANSCRIPTION_PROMPT = "Please transcribe this speech."
TOXICITY_PROMPT = (
    "Does this speech contain any toxic, harmful, hateful, offensive, or inappropriate language? "
    "Return your answer in JSON format as {{\"label\": 1 or 0, \"reason\": \"your explanation\"}}. "
    "Use 1 for toxic and 0 for non-toxic."
)

def read_csv_robust(path):
    """Read CSV with multiple encoding attempts."""
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    # Fallback: replace undecodable bytes
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

def transcribe_audio_meralion(model, processor, audio_path, prompt):
    """Transcribe or analyze audio using MERaLiON model."""
    try:
        audio, _ = librosa.load(audio_path, sr=16000)
        conversation = [[{"role": "user", "content": prompt}]]
        chat_prompt = processor.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=chat_prompt, audios=[audio])
        outputs = model.generate(**inputs, max_new_tokens=256)
        generated_ids = outputs[:, inputs['input_ids'].size(1):]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    except Exception as e:
        print(f"Error processing audio {audio_path}: {e}")
        return ""

def classify_toxicity_multiple(texts):
    """Run multiple toxicity classifiers on texts."""
    results = {}
    
    # 1. Toxic-BERT
    print("Running Toxic-BERT classification...")
    toxic_bert = pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        top_k=6,  # Return all 6 class scores for Toxic-BERT
        truncation=True,
        max_length=256,
        padding=True,
        batch_size=32,
        device=0 if torch.cuda.is_available() else -1,
    )
    toxic_bert_preds = toxic_bert(texts)
    results['toxic_bert_toxic'] = [
        int(any(label["score"] >= 0.5 for label in sample_scores))
        for sample_scores in tqdm(toxic_bert_preds, desc="Processing Toxic-BERT results")
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
    results['detoxify_toxic'] = detoxify_preds
    
    return results

def extract_meralion_toxicity_decision(response):
    """Extract binary toxicity decision from MERaLiON JSON response"""
    if not isinstance(response, str):
        return 0
    try:
        # Find and parse the first JSON object in the response
        json_start = response.find('{')
        json_end = response.find('}', json_start)
        if json_start != -1 and json_end != -1:
            data = json.loads(response[json_start:json_end+1])
            return int(data.get("label", 0) == 1)
        # Try to parse the whole response as JSON
        data = json.loads(response)
        return int(data.get("label", 0) == 1)
    except Exception:
        return 0

def clean_meralion_transcript(response):
    """Clean MERaLiON transcription response."""
    if not isinstance(response, str):
        return ""
    # Remove speaker tags like <Speaker1>: or <Speaker2>:
    cleaned = re.sub(r"<\s*speaker\d+\s*>:?", "", response, flags=re.IGNORECASE)
    # Remove any remaining tags like <...>
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    # Remove leading/trailing whitespace and redundant spaces/newlines
    cleaned = cleaned.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned

# Main execution
def main():
    # Load and filter metadata
    print(f"Loading metadata from: {META_PATH}")
    df = read_csv_robust(META_PATH)
    df = df[df["Dataset"] == "Common Voice"].reset_index(drop=True)
    print(f"Loaded {len(df)} Common Voice samples")

    # Load MERaLiON model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading MERaLiON model from: {REPO_ID} on {device}")
    processor = AutoProcessor.from_pretrained(REPO_ID, trust_remote_code=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        REPO_ID, use_safetensors=True, trust_remote_code=True, device_map=device
    )

    # Process each audio file
    print("Processing audio files...")
    transcripts = []
    toxicity_responses = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing audio"):
        try:
            file_path = find_audio_file(AUDIO_ROOT, row["Dataset"], row["FileName"])
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            transcripts.append("")
            toxicity_responses.append("")
            continue
        
        # Get transcription
        transcript_prompt = PROMPT_TEMPLATE.format(query=TRANSCRIPTION_PROMPT)
        transcript_raw = transcribe_audio_meralion(model, processor, file_path, transcript_prompt)
        transcript = clean_meralion_transcript(transcript_raw)
        transcripts.append(transcript)
        
        # Get toxicity analysis from MERaLiON
        toxicity_prompt = PROMPT_TEMPLATE.format(query=TOXICITY_PROMPT)
        toxicity_response = transcribe_audio_meralion(model, processor, file_path, toxicity_prompt)
        toxicity_responses.append(toxicity_response)

    # Use new column names
    df["pred_text"] = transcripts
    df["tox_json"] = toxicity_responses
    df["meralion_tox"] = [
        extract_meralion_toxicity_decision(resp) for resp in toxicity_responses
    ]

    # Run multiple toxicity classifiers on transcripts
    toxicity_results = classify_toxicity_multiple(df["pred_text"].tolist())
    for col, values in toxicity_results.items():
        # Map to new column names
        if col == "toxic_bert_toxic":
            df["bert_tox"] = values
        elif col == "detoxify_toxic":
            df["detoxy_tox"] = values

    # Compute WER/CER metrics
    print("Computing WER/CER metrics...")
    valid_pairs = [(ref, hyp) for ref, hyp in zip(df["text"], df["pred_text"]) 
                   if isinstance(ref, str) and isinstance(hyp, str)]
    
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
            f"Corpus WER: {corpus_metrics.wer:.4f}\n"
            f"Corpus CER: {corpus_metrics.cer:.4f}\n"
            f"Valid utterances: {len(valid_pairs)}/{len(df)}\n"
        )
    else:
        df["wer"] = [float('inf')] * len(df)
        df["cer"] = [float('inf')] * len(df)
        wer_summary = "No valid reference-hypothesis pairs found\n"

    # Generate classification reports
    print("Generating classification reports...")
    gold_labels = df["label2a"]
    reports = []
    for classifier in ["meralion_tox", "bert_tox", "detoxy_tox"]:
        if classifier in df.columns and df[classifier].notna().any():
            if classifier == "detoxy_tox" and df[classifier].isna().all():
                continue
            report = classification_report(
                gold_labels, df[classifier], 
                target_names=['Non-toxic', 'Toxic'], 
                output_dict=False
            )
            reports.append(f"\n{classifier.upper()} Classification Report:\n{report}")

    # Save results
    print(f"Saving results to: {RESULT_DIR}")
    with open(os.path.join(RESULT_DIR, f"{MODEL_NAME}_summary.txt"), "w", encoding="utf-8") as f:
        f.write("TOXICITY EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(wer_summary)
        f.write("\n")
        for report in reports:
            f.write(report)
            f.write("\n")

    # Prepare and save CSV
    output_columns = [
        "ref_text", "pred_text", "gold_tox", "meralion_tox", "tox_json", "bert_tox", "detoxy_tox", "wer", "cer"
    ]
    # Rename columns for output
    output_df = df.rename(columns={
        "text": "ref_text",
        "label2a": "gold_tox"
    })
    output_df[output_columns].to_csv(
        os.path.join(RESULT_DIR, f"{MODEL_NAME}_results.csv"), 
        index=False
    )

    # Print summary
    print("\nEVALUATION SUMMARY:")
    print("=" * 50)
    print(wer_summary)
    for report in reports:
        print(report)
    print(f"\nResults saved to:")
    print(f"- {RESULT_DIR}/{MODEL_NAME}_summary.txt")
    print(f"- {RESULT_DIR}/{MODEL_NAME}_results.csv")
    print("Evaluation completed!")

if __name__ == "__main__":
    main()