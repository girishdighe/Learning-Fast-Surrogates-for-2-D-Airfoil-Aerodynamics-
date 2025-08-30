# src/utils/paths.py
"""
Path resolution helpers.
- Read DATA_ROOT from config with optional environment override
- Build output directories safely
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import os


def resolve_data_root(cfg: Dict) -> Path:
    """
    Resolve data_root using (env override) or cfg["data_root"].
    """
    env = os.environ.get("DATA_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    root = cfg.get("data_root", None)
    if not root:
        raise KeyError("data_root not found in config and DATA_ROOT env not set.")
    return Path(root).expanduser().resolve()


def ensure_dir(p: Path | str) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_dir(base_out: Path | str, run_name: str) -> Path:
    return ensure_dir(Path(base_out) / "runs" / run_name)
