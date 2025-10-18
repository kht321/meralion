import csv
import json

input1_csv = "data/detoxy/metadata/test.csv"
output1_json = "data/detoxy/metadata/test.json"

result = []
with open(input1_csv, encoding="latin1") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["Dataset"].strip() == "Common Voice":
            result.append({
                "file": f'{row["FileName"].strip()}.WAV',
                "transcript": row["text"].strip(),
                "category": "Common Voice"
            })

with open(output1_json, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"Generated {output1_json} with {len(result)} entries.")


input2_csv = "data/detoxy/metadata/trigger_test.csv"
output2_json = "data/detoxy/metadata/trigger_test.json"

result = []
with open(input2_csv, encoding="latin1") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["Dataset"].strip() == "Common Voice":
            result.append({
                "file": f'{row["FileName"].strip()}.WAV',
                "transcript": row["text"].strip(),
                "category": "Common Voice"
            })

with open(output2_json, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"Generated {output2_json} with {len(result)} entries.")
