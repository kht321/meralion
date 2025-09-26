"""Device selection helpers."""

from __future__ import annotations

import torch


class Device:
    """Pick an appropriate torch device with sensible defaults."""

    def __init__(self) -> None:
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and getattr(mps_backend, "is_available", lambda: False)():
            self.type = "mps"
            if hasattr(mps_backend, "allow_tf32"):
                mps_backend.allow_tf32 = True
        elif torch.cuda.is_available():
            self.type = "cuda"
        else:
            self.type = "cpu"

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float16 if self.type in {"cuda", "mps"} else torch.float32

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.type


__all__ = ["Device"]
