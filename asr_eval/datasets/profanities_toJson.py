
import pandas as pd
import json
import re
import string

# Convert profanities metadata CSV to JSON
input_xlsx = "data/profanities/metadata.xlsx"
output_json = "data/profanities/metadata.json"

result = []
df = pd.read_excel(input_xlsx)

def clean_transcript(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Remove leading "model" + optional punctuation or spaces
    text = re.sub(r'^\s*model[\s:>,-]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(speaker1|speaker2)\b[:\s-]*', '', text, flags=re.IGNORECASE)

    # Remove all punctuation EXCEPT square brackets
    punct_to_remove = string.punctuation.replace('[', '').replace(']', '')
    text = re.sub(f"[{re.escape(punct_to_remove)}]", "", text)

    # Lowercase and normalize whitespace
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


for index, row in df.iterrows():
        transcript = clean_transcript(row.get("transcribed first 30s (meralion)", ""))
        if not isinstance(transcript, str):
            transcript = ""
        else:
            transcript = transcript.strip()

        # Use mp3 file label for filename
        file_label = row.get("mp3 file label", "").strip()
        toxic = row.get("Toxic", "")

        # Only add if file_label is present
        if file_label:
            result.append({
                "file": f"{file_label}.mp3",
                "transcript": transcript
            })

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"Generated {output_json} with {len(result)} entries.")
