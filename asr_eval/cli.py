"""Command line interface for running ASR evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audio import TARGET_SR, load_audio, resample_np
from .device import Device
from .models.meralion import MERaLiON
from .models.whisper import Whisper


ALIASES = {
    "meralion-2-10b": "MERaLiON/MERaLiON-2-10B",
    "meralion-2-3b": "MERaLiON/MERaLiON-2-3B",
    "whisper-small": "openai/whisper-small",
}


def build_model(name: str, device: Device):
    key = name.lower()
    if key in {"meralion", "meralion-2-10b"}:
        return MERaLiON(ALIASES["meralion-2-10b"], device)
    if key == "meralion-2-3b":
        return MERaLiON(ALIASES["meralion-2-3b"], device)
    if key in {"whisper", "whisper-small"}:
        return Whisper(ALIASES["whisper-small"], device)

    if "meralion" in key:
        return MERaLiON(name, device)
    return Whisper(name, device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ASR eval CLI: transcribe wav(s) with MERaLiON or Whisper",
    )
    parser.add_argument("inputs", nargs="+", help="audio file(s)")
    parser.add_argument("--model", default="whisper-small", help="alias or HF id")
    parser.add_argument("--json", dest="as_json", action="store_true", help="emit JSON lines")
    args = parser.parse_args()

    device = Device()
    print(f"device={device}")

    model = build_model(args.model, device)

    try:
        for path in args.inputs:
            wav, sr = load_audio(path)
            if sr != TARGET_SR:
                wav = resample_np(wav, sr, TARGET_SR)
                sr = TARGET_SR

            transcript = model.transcribe(wav, sr)
            if args.as_json:
                print(json.dumps({"path": path, "text": transcript}))
            else:
                print(f"{Path(path).name}: {transcript}")
    finally:
        model.close()


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()


__all__ = ["main", "build_model"]
