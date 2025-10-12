"""Convert Excel metadata (transcripts + filenames) to JSONL/CSV manifest.

Expected input Excel columns (case-sensitive):
- Filename : audio filename (e.g. test1.mp3)
- GT       : ground-truth transcription (text)
- meralion : model transcript (optional)
- Gender, Race, Age, ... : any metadata columns to include

Output manifest rows will contain at least:
- audio: {"path": "data/fairness/audio/test1.mp3"}
- audio_path: same path (legacy field)
- text: the ground-truth transcription (from GT)
and will copy any other columns from the Excel row.

Usage (PowerShell):
  python scripts/excel_to_manifest.py \
    --input results/fairness/metadata.xlsx \
    --audio-dir data/fairness/audio \
    --out data/manifests/fairness_manifest.jsonl

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def read_excel(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)


def make_manifest_row(row: Dict[str, Any], audio_dir: Path, filename_col: str = "Filename", text_col: str = "GT") -> Dict[str, Any]:
    fname = row.get(filename_col)
    if not isinstance(fname, str) or not fname:
        raise ValueError(f"Missing or invalid filename in row: {row}")

    # Support both absolute or just filename
    p = Path(fname)
    if p.is_absolute():
        rel_path = str(p)
    else:
        rel_path = str((audio_dir / p).as_posix())

    out: Dict[str, Any] = {}
    out["audio"] = {"path": rel_path}
    out["audio_path"] = rel_path

    # Map GT -> text
    text_val = row.get(text_col, "")
    out["text"] = text_val if text_val is not None else ""

    # Copy all other fields (including meralion, Gender, Race, Age, etc.)
    for k, v in row.items():
        if k in {filename_col, text_col}:
            continue
        out[k] = v if (v is not None and (not (isinstance(v, float) and pd.isna(v)))) else None

    return out


def write_jsonl(rows, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(rows, dest: Path) -> None:
    import csv

    dest.parent.mkdir(parents=True, exist_ok=True)
    # infer headers from union of all keys
    headers = set()
    for r in rows:
        headers.update(r.keys())
    headers = list(sorted(headers))

    with dest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (v if v is not None else "") for k, v in r.items()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/fairness/metadata.xlsx"), help="Excel input file")
    parser.add_argument("--audio-dir", type=Path, default=Path("data/fairness/audio"), help="Directory where audio files are located")
    parser.add_argument("--out", type=Path, default=Path("data/manifests/fairness_manifest.jsonl"), help="Output manifest JSONL path")
    parser.add_argument("--csv-out", type=Path, default=None, help="Optional CSV manifest output path")
    parser.add_argument("--filename-col", type=str, default="Filename", help="Column name for filename in Excel")
    parser.add_argument("--text-col", type=str, default="GT", help="Column name for ground-truth text in Excel")

    args = parser.parse_args()

    df = read_excel(args.input)
    rows = []
    for _, r in df.iterrows():
        row = r.to_dict()
        try:
            m = make_manifest_row(row, args.audio_dir, filename_col=args.filename_col, text_col=args.text_col)
            rows.append(m)
        except Exception as e:
            print(f"Skipping row due to error: {e}")

    write_jsonl(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out}")

    if args.csv_out:
        write_csv(rows, args.csv_out)
        print(f"Wrote CSV manifest to {args.csv_out}")


if __name__ == "__main__":
    main()
