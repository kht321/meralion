"""Audio utilities for ASR evaluation."""

from __future__ import annotations

import numpy as np
from pathlib import Path

try:
    import soundfile as sf
except Exception:  # pragma: no cover - optional dependency path
    sf = None


TARGET_SR = 16_000


def load_audio(path: str, max_duration_sec: float | None = None) -> tuple[np.ndarray, int]:
    """
    Load audio file with support for WAV and MP3 formats.

    Parameters
    ----------
    path : str
        Path to audio file (WAV or MP3)
    max_duration_sec : float | None
        If specified, truncate audio to this duration in seconds

    Returns
    -------
    tuple[np.ndarray, int]
        Audio samples (float32) and sample rate
    """
    if sf is None:
        raise RuntimeError("soundfile not installed")

    path_obj = Path(path)

    # Check if file is MP3 and needs conversion
    if path_obj.suffix.lower() == '.mp3':
        try:
            import subprocess
            import tempfile
            import os

            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
                tmp_path = tmp_wav.name

            # Convert MP3 to WAV using ffmpeg
            result = subprocess.run(
                ['ffmpeg', '-i', str(path), '-ar', str(TARGET_SR), '-ac', '1', '-y', tmp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30
            )

            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg conversion failed: {result.stderr.decode()}")

            # Load the converted WAV
            wav, sr = sf.read(tmp_path)

            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise RuntimeError(f"Failed to convert MP3 (ffmpeg required): {e}")
    else:
        # Load WAV directly
        wav, sr = sf.read(path)

    arr = np.asarray(wav, dtype=np.float32)

    if arr.ndim > 1 and arr.shape[-1] > 1:
        arr = arr.mean(axis=-1)
    elif arr.ndim > 1:
        arr = np.squeeze(arr, axis=-1)

    # Truncate to max duration if specified
    if max_duration_sec is not None and max_duration_sec > 0:
        max_samples = int(max_duration_sec * sr)
        if len(arr) > max_samples:
            arr = arr[:max_samples]

    return arr.astype(np.float32, copy=False), int(sr)


def resample_np(wav: np.ndarray, orig_sr: int, tgt_sr: int = TARGET_SR) -> np.ndarray:
    if orig_sr == tgt_sr:
        return wav.astype(np.float32, copy=False)

    if orig_sr <= 0:
        raise ValueError("invalid sample rate")

    n_src = wav.shape[0]
    n_tgt = max(int(round(n_src * tgt_sr / orig_sr)), 1)
    x = np.linspace(0, n_src - 1, num=n_src, dtype=np.float64)
    xi = np.linspace(0, n_src - 1, num=n_tgt, dtype=np.float64)

    return np.interp(xi, x, wav).astype(np.float32)


__all__ = ["TARGET_SR", "load_audio", "resample_np"]
