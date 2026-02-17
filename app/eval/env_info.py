from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass


def _run_git(args: list[str]) -> str | None:
    try:
        out = subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
        return out or None
    except Exception:
        return None


def environment_payload() -> dict[str, object]:
    sha = _run_git(["rev-parse", "HEAD"])
    dirty = bool(_run_git(["status", "--porcelain"]))
    runtime_mode = os.getenv("GR_RUNTIME_MODE", "cpu")
    cpu_device = os.getenv("GR_CPU_DEVICE", "")
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_sha": sha,
        "git_dirty": dirty,
        "runtime_mode": runtime_mode,
        "cpu_device": cpu_device,
    }

