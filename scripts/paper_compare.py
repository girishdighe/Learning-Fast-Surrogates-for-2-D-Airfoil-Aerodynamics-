#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper-style qualitative comparison: GT vs UNet/FNO/POD with optional error panels.
This is a robust rewrite aligned with qualitative_montages.py behavior.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch

# Matplotlib setup (headless friendly)
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update(
    {"font.family": "serif", "font.size": 11, "figure.dpi": 300, "savefig.bbox": "tight"}
)

from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- Project imports -----------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
from torch.utils.data import DataLoader
from src.data.dataset_airfrans import AirfransDataset
from src.data.collate import simple_collate
from src.models.unet import UNet2D
from src.models.fno import FNO2D
from src.models.pod import PODRegressor


# ------------------------ small helpers ---------------------------------------
def load_yaml(p: str):
    with open(p, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def select_device(pref: str) -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _unwrap(batch):
    """Return (first_item_dict, case_id_str)."""
    b = batch[0] if isinstance(batch, list) else batch
    meta = b.get("meta", {})
    if isinstance(meta, list):
        meta = meta[0] if meta else {}
    return b, str(meta.get("case_id", "unknown"))


def _panel_colorbar(ax, im, label):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.02)
    cb = ax.figure.colorbar(im, cax=cax)
    cb.ax.set_ylabel(label)


def _row_colorbar(fig, axes_row, im, label):
    cax = fig.add_axes(
        [
            axes_row[0].get_position().x1 + 0.01,
            axes_row[-1].get_position().y0,
            0.012,
            axes_row[0].get_position().height,
        ]
    )
    cb = fig.colorbar(im, cax=cax)
    cb.ax.set_ylabel(label)


# ------------------------ model building / loading ----------------------------
def build_model(kind: str, mcfg: dict, in_ch: int, H: int, W: int, device: torch.device):
    """Instantiate model with safe defaults that match your training."""
    if kind == "unet":
        model = UNet2D(
            in_channels=in_ch,
            out_channels=int(mcfg.get("out_channels", 3)),
            base_channels=int(mcfg.get("base_channels", 32)),
            depth=int(mcfg.get("depth", 4)),
            dropout=float(mcfg.get("dropout", 0.0)),
        )
    elif kind == "fno":
        modes = mcfg.get("modes", [16, 16])
        if isinstance(modes, int):
            modes = [modes, modes]
        model = FNO2D(
            in_channels=in_ch,
            out_channels=int(mcfg.get("out_channels", 3)),
            modes=(int(modes[0]), int(modes[1])),
            layers=int(mcfg.get("layers", 4)),
            hidden_channels=int(mcfg.get("hidden_channels", 64)),
        )
    elif kind == "pod":
        # IMPORTANT: this PODRegressor in your repo does NOT accept out_channels.
        modes = int(mcfg.get("modes", 64))
        basis = mcfg.get("basis_path") or (Path("outputs/pod_basis") / f"pod_basis_{modes}.npz")
        # Some repos expose a richer signature; fall back to minimal if needed.
        try:
            model = PODRegressor(
                in_channels=in_ch,
                H=H,
                W=W,
                modes=modes,
                basis_path=str(basis),
                hidden=int(mcfg.get("hidden", 128)),
                mlp_layers=mcfg.get("hidden_layers", [128, 128]),
            )
        except TypeError:
            model = PODRegressor(
                in_channels=in_ch,
                H=H,
                W=W,
                modes=modes,
                basis_path=str(basis),
            )
    else:
        raise ValueError(f"unknown model kind: {kind}")
    return model.to(device)


def load_ckpt_strict(model: torch.nn.Module, ckpt_path: Path) -> torch.nn.Module:
    """Load checkpoint the same way your working script did: strict model_state."""
    state = torch.load(str(ckpt_path), map_location="cpu")
    if "model_state" not in state:
        # try common alt key
        if "state_dict" in state:
            state = {"model_state": state["state_dict"]}
        elif isinstance(state, dict) and all(hasattr(v, "dtype") for v in state.values()):
            state = {"model_state": state}
        else:
            raise RuntimeError(f"Checkpoint at {ckpt_path} missing model_state.")
    model.load_state_dict(state["model_state"], strict=True)
    return model.eval()


def restore_model_from_run(kind: str, run_dir: str, in_ch: int, H: int, W: int, device):
    """Build model from config file and load its ckpt_best.pt strictly."""
    cfg_file = {
        "unet": "configs/unet.yaml",
        "fno": "configs/fno.yaml",
        "pod": "configs/pod.yaml",
    }[kind]
    mcfg = load_yaml(cfg_file).get("model", {})
    model = build_model(kind, mcfg, in_ch, H, W, device)
    ck = Path(run_dir) / "ckpt_best.pt"
    return load_ckpt_strict(model, ck)


# ------------------------ normalization helpers -------------------------------
def _load_mean_std_npz(p: Path):
    z = np.load(str(p))
    return z["mean"].astype(np.float32), z["std"].astype(np.float32)


def choose_dataset_invnorm(base_cfg: dict):
    """
    If base_cfg['normalizer_path'] is a *string path* to an npz with mean/std,
    return (mean, std). Otherwise return None.
    """
    ds_norm = base_cfg.get("normalizer_path", None)
    if isinstance(ds_norm, (str, os.PathLike)):
        try:
            m, s = _load_mean_std_npz(Path(ds_norm))
            return m, s
        except Exception as e:
            print(f"[WARN] dataset normalizer at {ds_norm} not usable: {e}")
    else:
        if ds_norm is not None:
            print(f"[WARN] ignoring dataset normalizer at {ds_norm}: expected path string.")
    return None


def load_run_out_norm(run_dir: str):
    """Return (mean, std) from run_dir/out_norm.npz, or None if missing/bad."""
    p = Path(run_dir) / "out_norm.npz"
    if p.exists():
        try:
            return _load_mean_std_npz(p)
        except Exception as e:
            print(f"[WARN] could not load run out_norm at {p}: {e}")
    return None


def inv_zscore(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Inverse z-score per-channel: (C,H,W) = arr*std + mean."""
    C = arr.shape[0]
    return arr * std[:C, None, None] + mean[:C, None, None]


# ------------------------ main ------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Paper-style montages: GT vs UNet/FNO/POD with optional error panels."
    )
    ap.add_argument("--base_cfg", type=str, default="configs/base.yaml")
    ap.add_argument("--dataset_cfg", type=str, default="configs/dataset_airfrans.yaml")
    ap.add_argument("--unet_run", type=str, required=True)
    ap.add_argument("--fno_run", type=str, required=True)
    ap.add_argument("--pod_run", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max_cases", type=int, default=4)
    ap.add_argument("--outdir", type=str, default="outputs/paper_compare")

    # plotting options
    ap.add_argument("--global_minmax", action="store_true")
    ap.add_argument("--cmap_field", type=str, default="turbo")
    ap.add_argument("--cmap_error", type=str, default="coolwarm")
    ap.add_argument(
        "--colorbar",
        type=str,
        default="panel",
        choices=["panel", "row", "single", "none"],
        help="where to draw colorbars",
    )
    ap.add_argument("--symmetric_error", action="store_true")

    args = ap.parse_args()

    base = load_yaml(args.base_cfg)
    dcfg = load_yaml(args.dataset_cfg)
    include = base.get("include_channels", ["sdf", "mask", "aoa", "u_inf"])
    device = select_device(args.device)

    # merge (dataset class in this repo expects a single cfg sometimes)
    cfg = {}
    cfg.update(base)
    cfg.update(dcfg)

    # Build dataset exactly like the working script did
    ds = AirfransDataset(
        cfg,
        cfg,
        split="test",
        precompute_npz=cfg.get("precompute_npz", None),
        include_channels=include,
        device=None,
    )

    # peek to get geometry
    print(f"[INFO] dataset shapes: Cin={ds[0]['inputs'].shape[0]} HxW={ds[0]['inputs'].shape[-2]}x{ds[0]['inputs'].shape[-1]}")
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=simple_collate)
    first = next(iter(loader))
    b0, _ = _unwrap(first)
    H, W = int(b0["inputs"].shape[-2]), int(b0["inputs"].shape[-1])
    in_ch = int(b0["inputs"].shape[1])

    # Models (strict load, same as working script)
    unet = restore_model_from_run("unet", args.unet_run, in_ch, H, W, device)
    fno = restore_model_from_run("fno", args.fno_run, in_ch, H, W, device)
    pod = restore_model_from_run("pod", args.pod_run, in_ch, H, W, device)

    # Normalization choices:
    # - If dataset normalizer path is a usable npz, inverse it for GT and all preds.
    # - Else: leave GT as-is, but inverse each model's preds using its own run's out_norm.npz if present.
    ds_inv = choose_dataset_invnorm(base)  # None or (mean,std)
    if ds_inv is None:
        print("[INFO] no usable dataset normalizer; will inverse each model with its run out_norm if available.")
    unet_out_norm = load_run_out_norm(args.unet_run)
    fno_out_norm = load_run_out_norm(args.fno_run)
    pod_out_norm = load_run_out_norm(args.pod_run)

    # prepare output
    ensure_dir(args.outdir)

    # Optionally compute global min/max for field rows (per channel) on GT (after inverse if ds_inv)
    gmin = gmax = None
    if args.global_minmax:
        mins, maxs = [], []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= args.max_cases:
                    break
                b, _ = _unwrap(batch)
                y = b["targets"][0].cpu().numpy()  # (3,H,W)
                if ds_inv is not None:
                    y = inv_zscore(y, *ds_inv)
                mins.append(y.min(axis=(1, 2)))
                maxs.append(y.max(axis=(1, 2)))
        gmin = np.stack(mins).min(axis=0)
        gmax = np.stack(maxs).max(axis=0)
        # re-initialize loader (was consumed)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=simple_collate)

    names = ["u", "v", "p"]
    cols = ["GT", "UNet", "FNO", "POD"]

    done = 0
    with torch.no_grad():
        for batch in loader:
            if done >= args.max_cases:
                break
            b, case_id = _unwrap(batch)
            x = b["inputs"].to(device)   # (1,Cin,H,W)
            y = b["targets"].to(device)  # (1,3,H,W)

            # raw predictions
            pu = unet(x)[0].detach().cpu().numpy()
            pf = fno(x)[0].detach().cpu().numpy()
            pp = pod(x)[0].detach().cpu().numpy()
            gt = y[0].detach().cpu().numpy()

            # inverse normalization strategy
            if ds_inv is not None:
                gt = inv_zscore(gt, *ds_inv)
                pu = inv_zscore(pu, *ds_inv)
                pf = inv_zscore(pf, *ds_inv)
                pp = inv_zscore(pp, *ds_inv)
            else:
                # Leave GT as provided by dataset; inverse each model with its own run stats if available.
                if unet_out_norm is not None:
                    pu = inv_zscore(pu, *unet_out_norm)
                if fno_out_norm is not None:
                    pf = inv_zscore(pf, *fno_out_norm)
                if pod_out_norm is not None:
                    pp = inv_zscore(pp, *pod_out_norm)

            # mask (optional) — if 'mask' is one of the inputs
            mask = None
            if "mask" in include:
                mask_idx = include.index("mask")
                mask = (b["inputs"][0, mask_idx].cpu().numpy() > 0.5).astype(np.float32)

            # ---------------- FIELDS PANEL ----------------
            mats = {"GT": gt, "UNet": pu, "FNO": pf, "POD": pp}
            fig, axes = plt.subplots(3, 4, figsize=(12, 7))
            last_im = None
            for r, ch in enumerate(names):
                vmin = gmin[r] if gmin is not None else min(mats[k][r].min() for k in cols)
                vmax = gmax[r] if gmax is not None else max(mats[k][r].max() for k in cols)
                for c, name in enumerate(cols):
                    ax = axes[r, c]
                    im = ax.imshow(
                        mats[name][r], origin="lower", cmap=args.cmap_field, vmin=vmin, vmax=vmax, interpolation="nearest"
                    )
                    last_im = im
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f"{ch} {name}")
                    if args.colorbar == "panel":
                        _panel_colorbar(ax, im, ch)
                if args.colorbar == "row":
                    _row_colorbar(fig, axes[r, :], last_im, names[r])
            if args.colorbar == "single":
                fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.75).ax.set_ylabel("value")
            fig.suptitle(f"Case {case_id}", y=0.99)
            fig.savefig(Path(args.outdir) / f"{case_id}_fields.png")
            plt.close(fig)

            # ---------------- ERRORS PANEL ----------------
            fig2, axes2 = plt.subplots(3, 3, figsize=(10, 7))
            last_im = None
            for r, ch in enumerate(names):
                if args.symmetric_error:
                    E = max(
                        np.nanmax(np.abs(pu[r] - gt[r])),
                        np.nanmax(np.abs(pf[r] - gt[r])),
                        np.nanmax(np.abs(pp[r] - gt[r])),
                    )
                    evmin, evmax = -E, E
                else:
                    evmin = evmax = None
                for c, (name, arr) in enumerate(zip(["UNet", "FNO", "POD"], [pu, pf, pp])):
                    err = arr[r] - gt[r]
                    if mask is not None:
                        err = np.where(mask > 0.5, err, np.nan)
                    ax = axes2[r, c]
                    im = ax.imshow(
                        err, origin="lower", cmap=args.cmap_error, vmin=evmin, vmax=evmax, interpolation="nearest"
                    )
                    last_im = im
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f"{ch} error {name}")
                    if args.colorbar == "panel":
                        _panel_colorbar(ax, im, "error")
                if args.colorbar == "row":
                    _row_colorbar(fig2, axes2[r, :], last_im, "error")
            if args.colorbar == "single":
                fig2.colorbar(last_im, ax=axes2.ravel().tolist(), shrink=0.75).ax.set_ylabel("error")
            fig2.suptitle(f"Errors for {case_id}", y=0.99)
            fig2.savefig(Path(args.outdir) / f"{case_id}_errors.png")
            plt.close(fig2)

            done += 1

    print(f"[OK] wrote montages to {args.outdir}")


if __name__ == "__main__":
    main()
