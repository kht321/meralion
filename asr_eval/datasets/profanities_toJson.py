
import pandas as pd
import json

# Convert profanities metadata CSV to JSON
input_xlsx = "data/profanities/metadata.xlsx"
output_json = "data/profanities/metadata.json"

result = []
df = pd.read_excel(input_xlsx)
for index, row in df.iterrows():
        transcript = row.get("transcribed first 30s (meralion)", "")
        if not isinstance(transcript, str):
            transcript = ""
        else:
            transcript = transcript.strip()

        # Use mp3 file label for filename
        file_label = row.get("mp3 file label", "").strip()
        toxic = row.get("Toxic", "")
        video = row.get("Video", "")

        # Only add if file_label is present
        if file_label:
            result.append({
                "file": f"{file_label}.mp3",
                "transcript": transcript,
                "toxic": int(toxic) if str(toxic).isdigit() else toxic,
                "video": video
            })

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"Generated {output_json} with {len(result)} entries.")
