# src/utils/io_airfrans.py
"""
Airfrans I/O helpers:
- List case files
- Read splits (train/val/test/default)
- Load per-case HDF5 fields with a configurable key map
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import csv

import numpy as np

try:
    import h5py  # type: ignore
except Exception as e:  # pragma: no cover
    h5py = None
    _h5_err = e


def _require_h5py():
    if h5py is None:
        raise ImportError(
            "h5py is not available. Install it with `pip install h5py`."
        ) from _h5_err


def list_cases(data_root: Path | str, pattern: str = "case*.h5") -> List[Path]:
    """
    Return sorted list of case H5 files under data_root matching pattern.
    """
    root = Path(data_root)
    cases = sorted(root.glob(pattern))
    return cases


def _csv_has_header(csv_path: Path) -> bool:
    with csv_path.open("r", newline="") as f:
        first = f.readline().strip()
    if not first:
        return False
    # heuristic: if first token contains non-digit letters -> treat as header
    tokens = [t.strip() for t in first.replace(",", " ").split()]
    return any(any(ch.isalpha() for ch in t) for t in tokens)


def read_split_csv(csv_path: Path) -> List[str]:
    """
    Reads a split CSV. Accepts files that either:
      - contain a header and a column with case ids or filenames
      - or are a simple single-column file of case ids/filenames
    Returns a list of *case ids* (without .h5).
    """
    ids: List[str] = []
    if not csv_path.exists():
        raise FileNotFoundError(f"Split file not found: {csv_path}")

    hdr = _csv_has_header(csv_path)
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        if hdr:
            header = next(reader, None)
        for row in reader:
            if not row:
                continue
            token = row[0].strip()
            if not token:
                continue
            token = token.replace(".h5", "")
            ids.append(token)
    return ids


def load_split_ids(
    data_root: Path | str,
    splits_cfg: Dict[str, str],
    split_name: str,
) -> List[str]:
    """
    Load case identifiers for a named split using a dataset config mapping.
    Example: splits_cfg = {"train": "splits/train.csv", ...}
    """
    root = Path(data_root)
    rel = splits_cfg.get(split_name)
    if rel is None:
        raise KeyError(f"Split name '{split_name}' not found in config.")
    csv_path = root / rel if "/" in rel else (root / "splits" / rel)
    return read_split_csv(csv_path)


def read_h5_case(
    h5_path: Path | str,
    keymap: Dict[str, str],
    keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Read fields from a single .h5 case using the provided key map.

    Parameters
    ----------
    h5_path : Path
        Path to caseXYZ.h5
    keymap : dict
        Map like {"coords":"coords","u":"u","v":"v","p":"p","AoA":"AoA","Uinf":"Uinf",...}
    keys : list[str] | None
        Subset of keys to read (in terms of logical names from keymap). If None, read all present.

    Returns
    -------
    data : dict
        Dictionary of numpy arrays / scalars.
    """
    _require_h5py()
    p = Path(h5_path)
    if not p.exists():
        raise FileNotFoundError(p)

    data: Dict[str, Any] = {}
    with h5py.File(p, "r") as h5:
        candidates = keys if keys is not None else list(keymap.keys())
        for logical in candidates:
            dsname = keymap.get(logical)
            if dsname is None:
                continue
            if dsname not in h5:
                # silently skip missing optional keys
                continue
            arr = h5[dsname][...]
            # Ensure numpy arrays (not h5py Dataset)
            if hasattr(arr, "shape"):
                data[logical] = np.asarray(arr)
            else:
                data[logical] = arr
    return data


def infer_bbox_from_coords(coords: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute (xmin, xmax, ymin, ymax) from an array of points either with shape (N,2)
    or (...,2) grid-like structure.
    """
    arr = np.asarray(coords)
    if arr.ndim >= 2 and arr.shape[-1] == 2:
        xy = arr.reshape(-1, 2)
        x, y = xy[:, 0], xy[:, 1]
        return float(x.min()), float(x.max()), float(y.min()), float(y.max())
    raise ValueError("coords must have last dimension size 2 (got shape {})".format(arr.shape))
