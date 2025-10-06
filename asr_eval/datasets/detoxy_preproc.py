"""
In this project, only the Common Voice portion of the DeToxy dataset is used.
Download Common Voice Corpus 6.1 from: https://commonvoice.mozilla.org/en/datasets
"""


import sys
from pathlib import Path
import pandas as pd
import numpy as np
import os
import shutil
from pydub import AudioSegment
import tarfile
from tqdm import tqdm
import soundfile as sf
import librosa
from io import StringIO

# === CONFIGURATION ===
DATASET_ROOT = Path("data")
COMMONVOICE_TAR = DATASET_ROOT / "detoxy" / "en.tar"
COMMONVOICE_CLIPS = "cv-corpus-6.1-2020-12-11/en/clips/"
PREPROCESSED_ROOT = DATASET_ROOT / "detoxy" / "audio"
SPLITS = {
    "train": DATASET_ROOT / "detoxy" / "metadata" / "train.csv",
    "valid": DATASET_ROOT / "detoxy" / "metadata" / "valid.csv",
    "test": DATASET_ROOT / "detoxy" / "metadata" / "test.csv",
    "trigger_test": DATASET_ROOT / "detoxy" / "metadata" / "trigger_test.csv"
}

# Ensure output split directories exist
for split in SPLITS:
    (PREPROCESSED_ROOT / split).mkdir(parents=True, exist_ok=True)

# === LOAD SPLIT CSVs AND SELECT COMMON VOICE FILENAMES ===
def safe_read_csv(path):
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, on_bad_lines='warn')
            return df
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    with open(path, "rb") as f:
        data = f.read()
    text = data.decode("utf-8", errors="replace")
    return pd.read_csv(StringIO(text))

print("Loading split CSVs...")
split_dfs = {split: safe_read_csv(csv_path) for split, csv_path in SPLITS.items()}

# Get Common Voice file names for each split
def get_cv_filenames(df):
    if 'Dataset' not in df.columns or 'FileName' not in df.columns:
        return []
    ds_col = df['Dataset'].astype(str)
    is_cv = ds_col.str.lower().str.contains('common') | (ds_col == 'Common Voice')
    return df[is_cv]['FileName'].dropna().unique().tolist()

print("Selecting Common Voice filenames from splits...")
cv_files_by_split = {split: get_cv_filenames(df) for split, df in split_dfs.items()}
all_cv_files = sorted(set(sum(cv_files_by_split.values(), [])))

# === EXTRACT REQUIRED MP3 FILES FROM TAR ===
TMP_EXTRACT = DATASET_ROOT / "commonvoice" / "tmp_extract"
TMP_WAV = DATASET_ROOT / "commonvoice" / "tmp_wav"
TMP_EXTRACT.mkdir(parents=True, exist_ok=True)
TMP_WAV.mkdir(parents=True, exist_ok=True)

if all_cv_files:
    print("Extracting required MP3 files from tar...")
    file_names = [COMMONVOICE_CLIPS + fname + ".mp3" for fname in all_cv_files]
    with tarfile.open(COMMONVOICE_TAR) as tar:
        members = [m for m in tar.getmembers() if m.name in file_names]
        if members:
            tar.extractall(members=members, path=TMP_EXTRACT)
        else:
            print("No matching Common Voice files found in tar.")

    # === CONVERT MP3 TO WAV ===
    print("Converting MP3 files to WAV...")
    clips_path = TMP_EXTRACT / COMMONVOICE_CLIPS
    for fname in os.listdir(clips_path):
        if not fname.endswith(".mp3"):
            continue
        stem = Path(fname).stem
        sound = AudioSegment.from_mp3(str(clips_path / fname))
        sound.export(str(TMP_WAV / (stem + ".wav")), format="wav")

    # Remove TMP_EXTRACT after MP3->WAV conversion
    print("Removing extracted MP3 files to save space...")
    shutil.rmtree(TMP_EXTRACT, ignore_errors=True)

    # === CONVERT WAV TO 16kHz PCM WAV ===
    print("Converting WAV files to 16kHz PCM WAV...")
    TMP_16K = DATASET_ROOT / "commonvoice" / "tmp_16k"
    TMP_16K.mkdir(parents=True, exist_ok=True)
    for fname in tqdm(os.listdir(TMP_WAV), desc="Converting to 16kHz"):
        src_file = TMP_WAV / fname
        dst_file = TMP_16K / fname
        try:
            y, sr = librosa.load(str(src_file), sr=None)
            y_16 = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sf.write(str(dst_file), y_16, 16000, 'PCM_16')
        except Exception as e:
            print(f"Error processing {src_file}: {e}")

    # Remove TMP_WAV after WAV->16kHz conversion
    print("Removing intermediate WAV files to save space...")
    shutil.rmtree(TMP_WAV, ignore_errors=True)

    # === MOVE FINAL FILES TO SPLIT FOLDERS ===
    print("Moving final files to split folders...")
    for split, files in cv_files_by_split.items():
        split_dir = PREPROCESSED_ROOT / split
        for fname in files:
            src = TMP_16K / (fname + ".wav")
            if src.exists():
                shutil.copy2(str(src), str(split_dir / (fname + ".wav")))
            else:
                print(f"Missing processed file for {fname}")

    # Remove TMP_16K after moving files
    print("Removing 16kHz intermediate files to save space...")
    shutil.rmtree(TMP_16K, ignore_errors=True)

else:
    print("No Common Voice files found in splits. Nothing to process.")

print("Preprocessing complete. Final files are in:", PREPROCESSED_ROOT)
