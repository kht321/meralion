"""Helpers to expand corruption grids specified in configs."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


def expand_grid(spec: Dict[str, Iterable]) -> List[Tuple[str, dict]]:
    """Return a list of (corruption name, metadata dict) entries."""
    grid: List[Tuple[str, dict]] = []
    for name, values in spec.items():
        if name == "none":
            grid.append(("none", {"severity": 0}))
            continue
        for value in values:
            grid.append((name, {"severity": value}))
    return grid


__all__ = ["expand_grid"]
