import os
from pathlib import Path
import torch
import librosa
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from sklearn.metrics import classification_report
import jiwer
import sys

try:
    from asr_eval.metrics.wer_cer import corpus_scores
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from asr_eval.metrics.wer_cer import corpus_scores

# File paths
model_name = "MERaLiON-2-3B"
audio_root = "data/detoxy/audio/test" 
meta_path = "data/detoxy/metadata/test.csv" 
result_dir = "results/toxicity"
os.makedirs(result_dir, exist_ok=True)

# Load metadata
print(f"Loading metadata from: {meta_path}")
def _read_csv(path):
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    # Fallback: replace undecodable bytes
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return pd.read_csv(f)
df = _read_csv(meta_path)
df = df[df["Dataset"] == "Common Voice"].reset_index(drop=True)

# Find audio file for Common Voice rows
def _find_audio(root: str, dataset: str, fname: str) -> str:
    exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    cands = []
    for ext in exts:
        cands.append(os.path.join(root, f"{fname}{ext}"))
        cands.append(os.path.join(root, dataset, f"{fname}{ext}"))
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Audio not found for {fname} under {root}")

# Load STT model
repo_id = "asr_eval/models_local/MERaLiON-2-3B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading STT model from: {repo_id}")
processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    repo_id, use_safetensors=True, trust_remote_code=True, device_map=device
)

prompt_template = "Instruction: {query} \nFollow the text instruction based on the following audio: <SpeechHere>"
transcribe_prompt = "Please transcribe this speech."

# Transcribe each audio file
print("Starting transcription...")
transcripts = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Transcribing"):
    try:
        file_path = _find_audio(audio_root, row["Dataset"], row["FileName"])
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        transcripts.append("")
        continue
    audio, _ = librosa.load(file_path, sr=16000)
    conversation = [[{"role": "user", "content": prompt_template.format(query=transcribe_prompt)}]]
    chat_prompt = processor.tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=chat_prompt, audios=[audio])
    outputs = model.generate(**inputs, max_new_tokens=256)
    generated_ids = outputs[:, inputs['input_ids'].size(1):]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    transcripts.append(response)
df["stt_transcript"] = transcripts

# Toxicity classification and evaluation
print("Running toxicity inference...")
toxicity_model = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    return_all_scores=True,
    function_to_apply="sigmoid",
    truncation=True,
)
preds = toxicity_model(df["stt_transcript"].tolist(), batch_size=32)
df["stt_toxicity"] = [
    int(any(label["score"] >= 0.5 for label in sample_scores))
    for sample_scores in tqdm(preds, total=len(preds), desc="Postprocess toxicity")
]
# Use 'label2a' as the gold label for Common Voice
report = classification_report(df["label2a"], df["stt_toxicity"], output_dict=False)

# WER calculation using corpus_scores and per-utterance WER
print("Computing WER/CER...")
scores = corpus_scores(df["text"], df["stt_transcript"])
df["stt_wer"] = [
    jiwer.wer(ref if isinstance(ref, str) else "", hyp if isinstance(hyp, str) else "")
    for ref, hyp in zip(df["text"], df["stt_transcript"])
]
wer_summary = (
    f"Corpus WER: {scores.wer:.4f}\n"
    f"Corpus CER: {scores.cer:.4f}\n"
    f"Num utterances: {scores.n_utts}\n"
)

# Print and save results
print(report)
print(wer_summary)
print(f"Saving results to: {result_dir}")
with open(os.path.join(result_dir, f"{model_name}_class_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)
    f.write("\n")
    f.write(wer_summary)

# Save CSV (use 'text' and 'label2a' for Common Voice)
df[["text", "stt_transcript", "label2a", "stt_toxicity", "stt_wer"]].to_csv(
    os.path.join(result_dir, f"{model_name}_results.csv"), index=False
)
print("Completed!")
