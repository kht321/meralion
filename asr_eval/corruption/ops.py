"""Audio corruption operators for robustness evaluation."""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

TARGET_SR = 16_000


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(None if seed is None else int(seed))


def none(wav: np.ndarray, sr: int, **kwargs) -> np.ndarray:
    """Return the waveform unchanged (cast to float32)."""
    _ = kwargs  # unused
    return wav.astype(np.float32, copy=False)


def noise_snr_db(
    wav: np.ndarray,
    sr: int,
    *,
    snr_db: float,
    seed: int | None = None,
    **kwargs,
) -> np.ndarray:
    """Additive white noise at a target SNR (in dB)."""
    _ = kwargs
    generator = _rng(seed)
    x = wav.astype(np.float32, copy=False)
    power_signal = np.mean(x**2, dtype=np.float64) + 1e-12
    power_noise = power_signal / np.power(10.0, snr_db / 10.0)
    noise = generator.normal(0.0, np.sqrt(power_noise, dtype=np.float64), size=x.shape)
    noisy = x + noise.astype(np.float32)
    return np.clip(noisy, -1.0, 1.0)


def speed(wav: np.ndarray, sr: int, *, factor: float, **kwargs) -> np.ndarray:
    """Apply speed perturbation via resampling."""
    _ = kwargs
    factor = float(max(factor, 1e-6))
    n_src = wav.shape[0]
    n_tgt = max(int(round(n_src / factor)), 1)
    x = np.linspace(0, n_src - 1, num=n_src, dtype=np.float64)
    xi = np.linspace(0, n_src - 1, num=n_tgt, dtype=np.float64)
    y = np.interp(xi, x, wav).astype(np.float32)
    return y


def pitch_semitones(
    wav: np.ndarray,
    sr: int,
    *,
    semitones: float,
    **kwargs,
) -> np.ndarray:
    """Approximate pitch shift using resample + time-stretch."""
    _ = kwargs
    factor = np.power(2.0, float(semitones) / 12.0)
    n_src = wav.shape[0]
    n_mid = max(int(round(n_src / max(factor, 1e-6))), 1)
    x = np.linspace(0, n_src - 1, num=n_src, dtype=np.float64)
    xi = np.linspace(0, n_src - 1, num=n_mid, dtype=np.float64)
    mid = np.interp(xi, x, wav).astype(np.float32)
    xi2 = np.linspace(0, n_mid - 1, num=n_src, dtype=np.float64)
    y = np.interp(xi2, np.arange(n_mid, dtype=np.float64), mid).astype(np.float32)
    return y


def reverb_decay(
    wav: np.ndarray,
    sr: int,
    *,
    decay: float,
    seed: int | None = None,
    **kwargs,
) -> np.ndarray:
    """Simple exponential decay reverberation."""
    _ = kwargs
    decay = max(float(decay), 1e-3)
    n = max(int(0.3 * sr), 1)
    t = np.linspace(0.0, 1.0, num=n, dtype=np.float64)
    impulse = np.exp(-t / decay).astype(np.float32)
    impulse /= float(np.sum(impulse) + 1e-12)
    y = np.convolve(wav.astype(np.float32), impulse, mode="full")[: wav.shape[0]]
    return np.clip(y, -1.0, 1.0)


def clipping_ratio(
    wav: np.ndarray,
    sr: int,
    *,
    ratio: float,
    **kwargs,
) -> np.ndarray:
    """Hard-clip the waveform at +/- ratio."""
    _ = kwargs
    threshold = float(ratio)
    return np.clip(wav.astype(np.float32, copy=False), -threshold, threshold)


REGISTRY: Dict[str, Callable[..., np.ndarray]] = {
    "none": none,
    "noise_snr_db": noise_snr_db,
    "speed": speed,
    "pitch_semitones": pitch_semitones,
    "reverb_decay": reverb_decay,
    "clipping_ratio": clipping_ratio,
}

__all__ = ["TARGET_SR", "REGISTRY", "none", "noise_snr_db", "speed", "pitch_semitones", "reverb_decay", "clipping_ratio"]
