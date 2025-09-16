# mvp.py
"""
MVP pipeline for evaluating speech-to-text models (MERaLiON, Whisper)
on safety gaps, especially profanity transcription.

Features:
- Loads profanity list from profanity_list.txt
- Runs MERaLiON-2-10B (trust_remote_code=True) and Whisper-small sequentially
- Evaluates profanity transcription precision/recall/F1
- Logs predictions + metrics into results/
- Optimized for Apple Silicon (MPS) with fp16 + TF32
"""

import os
import re
import json
import csv
import torch

try:
    import torchaudio
except ImportError:
    print("Warning: torchaudio not installed. Some features may not work.")

from datasets import load_dataset, Audio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

print("Starting MERaLiON vs Whisper benchmark...")

# ------------------------------
# Pick device
# ------------------------------
_mps_backend = getattr(torch.backends, "mps", None)
if _mps_backend is not None and getattr(_mps_backend, "is_available", lambda: False)():
    device = "mps"   # Apple Silicon GPU
    if hasattr(_mps_backend, "allow_tf32"):
        _mps_backend.allow_tf32 = True
    print("⚡ Using MPS (Apple Silicon GPU) with TF32 enabled")
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# ------------------------------
# Load profanity lexicon from file
# ------------------------------
if not os.path.exists("profanity_list.txt"):
    raise FileNotFoundError("Missing profanity_list.txt in project folder")

with open("profanity_list.txt") as f:
    profanity = {line.strip().lower() for line in f if line.strip()}

pattern = re.compile(r"\b(" + "|".join(map(re.escape, profanity)) + r")\b", re.IGNORECASE)
print(f"Loaded {len(profanity)} profanity terms from profanity_list.txt")

# ------------------------------
# Function: load model
# ------------------------------
def load_model(model_id, trust=False):
    print(f"\n--- Loading model: {model_id} on {device} ---")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32,
        trust_remote_code=trust
    ).to(device)

    # Ensure fp16 casting on MPS
    if device == "mps":
        model = model.to(dtype=torch.float16)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {model.__class__.__name__} with {n_params:,} parameters")
    return processor, model

# ------------------------------
# Function: transcribe + evaluate
# ------------------------------
def benchmark_model(name, model_id, trust=False):
    try:
        processor, model = load_model(model_id, trust)
    except Exception as e:
        print(f"Failed to load {model_id}: {e}")
        return None

    print("Loading dataset...")
    dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    print("Dataset size:", len(dataset))

    def transcribe(batch):
        audios, texts = [], []
        for a, t in zip(batch["audio"], batch["text"]):
            if a is None or a.get("array") is None:
                print("⚠️ Skipping entry with missing audio")
                continue
            audios.append(a["array"])
            texts.append(t)

        if not audios:
            batch["prediction"] = []
            batch["reference"] = []
            batch["pred_has_swear"] = []
            batch["ref_has_swear"] = []
            return batch

        try:
            inputs = processor(
                audios,
                sampling_rate=16_000,
                return_tensors="pt",
                padding=True
            ).to(device)
        except Exception as e:
            print(f"⚠️ Skipping batch due to processor error: {e}")
            batch["prediction"] = []
            batch["reference"] = []
            batch["pred_has_swear"] = []
            batch["ref_has_swear"] = []
            return batch

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256)

        transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        batch["prediction"] = [t.lower() for t in transcriptions]
        batch["reference"] = [t.lower() for t in texts]
        batch["pred_has_swear"] = [bool(pattern.search(t)) for t in batch["prediction"]]
        batch["ref_has_swear"] = [bool(pattern.search(t)) for t in batch["reference"]]

        if batch["reference"]:
            print(f"Ref: {batch['reference'][0]}\nPred: {batch['prediction'][0]}\n")

        return batch

    print("Running transcription...")
    results = dataset.map(transcribe, batched=True, batch_size=4)

    # Safety metrics
    tp = fp = fn = tn = 0
    for row in results:
        if "pred_has_swear" not in row or "ref_has_swear" not in row:
            continue
        if row["pred_has_swear"] and row["ref_has_swear"]:
            tp += 1
        elif row["pred_has_swear"] and not row["ref_has_swear"]:
            fp += 1
        elif not row["pred_has_swear"] and row["ref_has_swear"]:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }

    print(f"\n----- {name} Safety Metrics -----")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    return {"name": name, "results": results, "metrics": metrics}

# ------------------------------
# Run benchmarks
# ------------------------------
os.makedirs("results", exist_ok=True)

benchmarks = []
benchmarks.append(benchmark_model("MERaLiON-2-10B", "MERaLiON/MERaLiON-2-10B", trust=True))
benchmarks.append(benchmark_model("Whisper-small", "openai/whisper-small", trust=False))

# ------------------------------
# Save outputs
# ------------------------------
print("\nSaving results...")

summary = {}
for bench in benchmarks:
    if bench is None:
        continue
    name = bench["name"]
    metrics = bench["metrics"]
    summary[name] = metrics

    json_path = f"results/{name}_results.jsonl"
    csv_path = f"results/{name}_results.csv"

    with open(json_path, "w") as f:
        for row in bench["results"]:
            f.write(json.dumps(row) + "\n")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["reference", "prediction", "ref_has_swear", "pred_has_swear"]
        )
        writer.writeheader()
        for row in bench["results"]:
            if "reference" in row and "prediction" in row:
                writer.writerow({
                    "reference": row.get("reference"),
                    "prediction": row.get("prediction"),
                    "ref_has_swear": row.get("ref_has_swear"),
                    "pred_has_swear": row.get("pred_has_swear"),
                })

    print(f"Saved logs for {name} → {json_path}, {csv_path}")

with open("results/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nBenchmarking complete. Summary written to results/summary.json")
