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

import numpy as np
import torch

try:
    import soundfile as sf  # optional fallback if HF Audio fails
except ImportError:
    sf = None

try:
    import torchaudio
except ImportError:
    torchaudio = None
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
def benchmark_model(name, model_id, trust=False, decoder_prompt_text=None):
    try:
        processor, model = load_model(model_id, trust)
    except Exception as e:
        print(f"Failed to load {model_id}: {e}")
        return None

    print("Loading dataset...")
    dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    print("Dataset size:", len(dataset))

    feature_extractor = getattr(processor, "feature_extractor", None)
    tokenizer = getattr(processor, "tokenizer", None)

    whisper_like = False
    padding_mode = None
    max_frames = None
    if feature_extractor is not None:
        whisper_like = feature_extractor.__class__.__name__.lower().startswith("whisper")
        max_frames = getattr(feature_extractor, "nb_max_frames", None)
        if whisper_like:
            padding_mode = "max_length"

    model_dtype = next(model.parameters()).dtype

    def transcribe(batch):
        predictions, references = [], []
        pred_has_swear, ref_has_swear = [], []

        first_logged = None

        for idx, audio_entry in enumerate(batch["audio"]):
            text_entry = batch.get("text", [""] * len(batch["audio"]))[idx]
            reference_text = (text_entry or "").lower()
            references.append(reference_text)
            ref_has_swear.append(bool(pattern.search(reference_text)))

            if audio_entry is None or audio_entry.get("array") is None:
                print(f"⚠️ [{name}] sample {idx}: missing audio array")
                predictions.append("")
                pred_has_swear.append(False)
                continue

            audio_np = np.asarray(audio_entry["array"], dtype=np.float32)
            if audio_np.ndim > 1:
                audio_np = np.squeeze(audio_np)

            if audio_np.size == 0:
                print(f"⚠️ [{name}] sample {idx}: empty audio array")
                predictions.append("")
                pred_has_swear.append(False)
                continue

            inputs_kwargs = {
                "audio": audio_np,
                "sampling_rate": 16_000,
                "return_tensors": "pt",
            }
            if decoder_prompt_text is not None and name.startswith("MERaLiON"):
                inputs_kwargs["text"] = decoder_prompt_text

            try:
                encoded = processor(**inputs_kwargs)
            except Exception as e:
                print(f"⚠️ [{name}] sample {idx}: processor failure: {e}")
                predictions.append("")
                pred_has_swear.append(False)
                continue

            if not isinstance(encoded, dict):
                try:
                    encoded = dict(encoded)
                except Exception:
                    encoded = encoded.data if hasattr(encoded, "data") else {}

            if "input_features" not in encoded:
                print(f"⚠️ [{name}] sample {idx}: no input_features from processor")
                predictions.append("")
                pred_has_swear.append(False)
                continue

            prepared_inputs = {}
            for key, value in encoded.items():
                if value is None:
                    continue
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value)
                if isinstance(value, torch.Tensor):
                    if key == "input_features":
                        value = value.to(device=device, dtype=model_dtype)
                    elif key.endswith("ids"):
                        value = value.to(device=device, dtype=torch.long)
                    else:
                        value = value.to(device=device)
                    prepared_inputs[key] = value

            if name.startswith("MERaLiON") and decoder_prompt_text is not None and "decoder_input_ids" not in prepared_inputs:
                try:
                    prompt_ids = processor.tokenizer(decoder_prompt_text, return_tensors="pt").input_ids
                    prepared_inputs["decoder_input_ids"] = prompt_ids.to(device=device, dtype=torch.long)
                except Exception:
                    pass

            generation_kwargs = {"max_new_tokens": 256}

            if whisper_like and hasattr(processor, "get_decoder_prompt_ids"):
                try:
                    forced_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
                    generation_kwargs["forced_decoder_ids"] = forced_ids
                except Exception:
                    pass

            try:
                with torch.no_grad():
                    generated_ids = model.generate(**prepared_inputs, **generation_kwargs)
                transcripts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                normalized = (transcripts[0] if transcripts else "").lower()
            except Exception as e:
                print(f"⚠️ [{name}] sample {idx}: generation failure: {e}")
                predictions.append("")
                pred_has_swear.append(False)
                continue

            predictions.append(normalized)
            pred_has_swear.append(bool(pattern.search(normalized)))

            if first_logged is None and normalized:
                first_logged = (reference_text, normalized)

        if first_logged is not None:
            ref_text, pred_text = first_logged
            print(f"Ref: {ref_text}\nPred: {pred_text}\n")

        return {
            "prediction": predictions,
            "reference": references,
            "pred_has_swear": pred_has_swear,
            "ref_has_swear": ref_has_swear,
        }

    print("Running transcription...")
    results = dataset.map(transcribe, batched=True, batch_size=4)

    # Safety metrics
    tp = fp = fn = tn = 0
    for row in results:
        if "pred_has_swear" not in row or "ref_has_swear" not in row:
            continue
        prediction_text = (row.get("prediction") or "").strip()
        if not prediction_text:
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
benchmarks.append(benchmark_model("MERaLiON-2-10B", "MERaLiON/MERaLiON-2-10B", trust=True, decoder_prompt_text=""))
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

    def json_safe(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.generic,)):
            return value.item()
        if isinstance(value, list):
            return [json_safe(v) for v in value]
        if isinstance(value, dict):
            cleaned = {}
            for k, v in value.items():
                if k == "array":
                    continue
                cleaned[k] = json_safe(v)
            return cleaned
        return value

    with open(json_path, "w") as f:
        for row in bench["results"]:
            safe_row = {k: json_safe(v) for k, v in row.items()}
            f.write(json.dumps(safe_row) + "\n")

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
