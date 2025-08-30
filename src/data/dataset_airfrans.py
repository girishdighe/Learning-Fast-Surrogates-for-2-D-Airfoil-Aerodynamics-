# src/data/dataset_airfrans.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.utils.io_airfrans import load_split_ids, read_h5_case
from src.utils.paths import resolve_data_root
from src.features.conditioning import broadcast_scalar_to_grid
from src.utils.normalization import Normalizer

try:
    import torch
    from torch.utils.data import Dataset
except Exception as e:
    raise ImportError("This module requires PyTorch. Install with `pip install torch`.") from e


def _load_precomp(precompute_npz: Optional[str]):
    if not precompute_npz: return None, None, None, None, None
    p = Path(precompute_npz)
    if not p.exists(): return None, None, None, None, None
    npz = np.load(p)
    X = npz["X"].astype(np.float32)
    Y = npz["Y"].astype(np.float32)
    sdf0 = npz["sdf"].astype(np.float32) if "sdf" in npz else None
    mask0 = npz["mask"].astype(np.float32) if "mask" in npz else None
    bbox = npz["bbox"].astype(np.float32) if "bbox" in npz else None
    return X, Y, sdf0, mask0, bbox

def _make_norm_grid(H: int, W: int):
    xs = np.linspace(0.0, 1.0, W, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, H, dtype=np.float32)
    return np.meshgrid(xs, ys, indexing="xy")


class AirfransDataset(Dataset):
    """
    Grid-based Airfrans:
      inputs  : [sdf, mask, aoa, u_inf]  (subset controlled by include_channels)
      targets : [u, v, p]
    """

    def __init__(self, cfg: Dict, dataset_cfg: Dict, split: str = "train",
                 precompute_npz: Optional[str] = None,
                 include_channels: Optional[List[str]] = None,
                 device: Optional[str] = None):
        super().__init__()
        self.device = device
        self.cfg = cfg
        self.dcfg = dataset_cfg
        self.data_root = resolve_data_root(cfg)
        self.splits_cfg = self.dcfg["splits"]
        self.keymap = self.dcfg["h5_keys"]
        self.case_ids: List[str] = load_split_ids(self.data_root, self.splits_cfg, split)
        if not self.case_ids:
            raise RuntimeError(f"No cases found for split '{split}'.")

        # precomputed grid/mask/sdf
        X0, Y0, self.sdf0, self.mask0, self.bbox = _load_precomp(precompute_npz)
        self.X, self.Y = X0, Y0

        # Peek size
        first = self._load_case_raw(self.case_ids[0])
        uvp = first.get("uvp", None)
        sdf = first.get("sdf", None)
        if uvp is not None:
            _, H, W = uvp.shape
        elif sdf is not None:
            H, W = sdf.shape
        else:
            raise RuntimeError("Cannot determine grid size from uvp/sdf.")
        if self.X is None or self.Y is None:
            self.X, self.Y = _make_norm_grid(H, W)
            self.bbox = np.array([0.0,1.0,0.0,1.0], dtype=np.float32)
        self.H, self.W = self.Y.shape[0], self.X.shape[1]

        # channels & normalization
        self.include_channels = include_channels or ["sdf","mask","aoa","u_inf"]
        norm_path = cfg.get("normalizer_path", None)
        # NEW (robust): ignore missing/invalid dataset normalizers for eval
        self.norm = None
        cand = None
        if isinstance(self.dcfg, dict):
            # accept any of these keys if present
            cand = self.dcfg.get("norm_path") or self.dcfg.get("normalization") or self.dcfg.get("norm")
        if cand:
            try:
                m = Normalizer.load(cand)
                # sanity: must expose mean/std
                mean = getattr(getattr(m, "state", None), "mean", None)
                std  = getattr(getattr(m, "state", None), "std",  None)
                if mean is not None and std is not None:
                    self.norm = m
                else:
                    print(f"[WARN] dataset normalizer at {cand} has wrong structure; ignoring.")
            except Exception as e:
                print(f"[WARN] ignoring dataset normalizer at {cand}: {e}")


        # optional RAM cache
        self._cache_enabled = False
        self._mem = None

    def enable_cache(self, cache: bool = True):
        self._cache_enabled = bool(cache)
        self._mem = None
        if not cache: return
        self._mem = []
        for cid in self.case_ids:
            self._mem.append(self._load_case_raw(cid))

    def __len__(self) -> int: return len(self.case_ids)

    def _load_case_raw(self, case_id: str) -> Dict:
        p = self.data_root / f"{case_id}.h5"
        if not p.exists():
            p = self.data_root / (case_id if case_id.endswith(".h5") else f"{case_id}.h5")
        want = ["uvp","sdf","valid","aoa","u_inf","re"]
        data = read_h5_case(p, self.keymap, keys=want)
        if "uvp" in data: data["uvp"] = np.asarray(data["uvp"]).astype(np.float32)
        if "sdf" in data: data["sdf"] = np.asarray(data["sdf"]).astype(np.float32)
        if "valid" in data: data["valid"] = (np.asarray(data["valid"]) > 0).astype(np.float32)
        if "aoa" in data: data["aoa"] = float(np.asarray(data["aoa"]))
        if "u_inf" not in data or data["u_inf"] is None: data["u_inf"] = 1.0
        return data

    def _build_inputs(self, data: Dict) -> np.ndarray:
        chans: List[np.ndarray] = []
        sdf = self.sdf0 if self.sdf0 is not None else data.get("sdf", np.zeros((self.H,self.W), np.float32))
        mask = self.mask0 if self.mask0 is not None else data.get("valid", np.ones((self.H,self.W), np.float32))
        for name in self.include_channels:
            if name == "sdf":  chans.append(sdf.astype(np.float32))
            elif name == "mask": chans.append(mask.astype(np.float32))
            elif name == "aoa": chans.append(broadcast_scalar_to_grid(float(data.get("aoa",0.0)), self.H, self.W))
            elif name == "u_inf": chans.append(broadcast_scalar_to_grid(float(data.get("u_inf",1.0)), self.H, self.W))
            else: chans.append(np.zeros((self.H,self.W), np.float32))  # unknown channel -> zeros
        x = np.stack(chans, axis=0).astype(np.float32)
        if self.norm is not None:
            x = self.norm.transform_inputs(x, self.include_channels)
        return x

    def _build_targets(self, data: Dict) -> np.ndarray:
        if "uvp" not in data:
            raise KeyError("This dataset requires 'uvp' (3,H,W).")
        y = data["uvp"]
        if y.shape[1:] != (self.H, self.W):
            raise ValueError(f"uvp shape {y.shape} does not match grid {(self.H,self.W)}")
        if self.norm is not None:
            y = self.norm.transform_targets(y)
        return y.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cid = self.case_ids[idx]
        data = self._mem[idx] if (self._cache_enabled and self._mem is not None) else self._load_case_raw(cid)
        x = self._build_inputs(data)
        y = self._build_targets(data)
        meta = {
            "case_id": cid,
            "AoA": float(data.get("aoa", 0.0)),
            "Uinf": float(data.get("u_inf", 1.0)),
            "bbox": self.bbox.copy() if self.bbox is not None else np.array([0,1,0,1], dtype=np.float32),
            "include_channels": self.include_channels,
        }
        return {
            "inputs": torch.from_numpy(x),
            "targets": torch.from_numpy(y),
            "X": torch.from_numpy(self.X),
            "Y": torch.from_numpy(self.Y),
            "meta": meta,
        }
