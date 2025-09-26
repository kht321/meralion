"""Audio utilities for ASR evaluation."""

from __future__ import annotations

import numpy as np

try:
    import soundfile as sf
except Exception:  # pragma: no cover - optional dependency path
    sf = None


TARGET_SR = 16_000


def load_audio(path: str) -> tuple[np.ndarray, int]:
    if sf is None:
        raise RuntimeError("soundfile not installed")

    wav, sr = sf.read(path)
    arr = np.asarray(wav, dtype=np.float32)

    if arr.ndim > 1 and arr.shape[-1] > 1:
        arr = arr.mean(axis=-1)
    elif arr.ndim > 1:
        arr = np.squeeze(arr, axis=-1)

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
