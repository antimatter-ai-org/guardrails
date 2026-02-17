from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_weights(path: str | Path) -> dict[str, float]:
    p = Path(path).expanduser().resolve()
    payload = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    weights = payload.get("weights") or {}
    if not isinstance(weights, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in weights.items():
        try:
            out[str(k).strip().lower()] = float(v)
        except Exception:
            continue
    return out

