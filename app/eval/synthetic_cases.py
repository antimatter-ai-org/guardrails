from __future__ import annotations

from pathlib import Path


def default_synthetic_case_dir() -> Path:
    return Path(".local") / "synthetic_cases"


def list_local_cases(base_dir: str | Path | None = None) -> list[str]:
    root = Path(base_dir) if base_dir is not None else default_synthetic_case_dir()
    if not root.exists():
        return []
    return sorted(path.name for path in root.glob("*.txt") if path.is_file())


def load_local_case(case_name: str, *, base_dir: str | Path | None = None) -> str:
    root = Path(base_dir) if base_dir is not None else default_synthetic_case_dir()
    path = root / case_name
    return path.read_text(encoding="utf-8")
