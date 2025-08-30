#!/usr/bin/env python3
import argparse, math, sys, importlib.util
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ----- paths -----
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT
SRC  = REPO / "src"
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
if str(SRC)  not in sys.path: sys.path.insert(0, str(SRC))

# ----- import dataset directly from file to avoid package path issues -----
def _import_from_file(py_path: Path, symbol: str):
    spec = importlib.util.spec_from_file_location(py_path.stem, str(py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return getattr(mod, symbol)

AirfransDataset = _import_from_file(REPO / "src" / "data" / "dataset_airfrans.py", "AirfransDataset")

# ----- models -----
from src.models.fno.fno2d import FNO2D
from src.models.unet.unet2d import UNet2D
try:
    from src.models.pod.pod import PODRegressor
except Exception:
    from src.models.pod.pod_head import PODRegressor

# ==================== helpers ====================
def crop2d(t: torch.Tensor, c: int) -> torch.Tensor:
    if c <= 0: return t
    return t[..., c:-c, c:-c]

def cdx(x):  # central diff x
    xp = F.pad(x, (1,1,0,0), mode="replicate")
    return 0.5 * (xp[..., 2:] - xp[..., :-2])

def cdy(x):  # central diff y
    xp = F.pad(x, (0,0,1,1), mode="replicate")
    return 0.5 * (xp[..., 2:, :] - xp[..., :-2, :])

def vorticity(u, v):  # dv/dx - du/dy
    return cdx(v) - cdy(u)

def divergence(u, v):  # du/dx + dv/dy
    return cdx(u) + cdy(v)

def edge_weight(y_u, alpha: float, gain: float):
    g = torch.sqrt(cdx(y_u)**2 + cdy(y_u)**2)
    m = g.mean().clamp_min(1e-6)
    return 1.0 + gain * torch.sigmoid(alpha * (g / m))

def wmean(a, w):
    return (a if w is None else a * w).mean()

# ----- run normalizer -----
def load_out_norm(run_dir: Path):
    z = np.load(run_dir / "out_norm.npz")
    mean = torch.from_numpy(z["mean"].astype(np.float32)).view(1, -1, 1, 1)
    std  = torch.from_numpy(z["std"].astype(np.float32)).view(1, -1, 1, 1)
    return mean, std

def inv_targets(pred_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    return pred_norm * std + mean

# ----- UNet base inference (so weights fit) -----
def _infer_unet_base(sd: dict) -> int:
    key = next((k for k in sd.keys() if k.endswith("inc.0.weight")), None)
    if key is None:
        raise RuntimeError("UNet checkpoint missing inc.0.weight")
    return int(sd[key].shape[0])

# ==================== POD basis (robust) ====================
def _load_basis_matrix(path: Path) -> np.ndarray:
    npz = np.load(path, allow_pickle=True)
    preferred = ("B", "basis", "Phi", "phi", "U", "V", "arr_0")
    for k in preferred:
        if k in npz:
            arr = np.array(npz[k])
            if arr.ndim == 2: return arr
    for k in npz.files:
        arr = np.array(npz[k])
        if arr.ndim == 2: return arr
    raise KeyError(f"{path} has no 2D array suitable as POD basis")

def _resolve_pod_basis(run_dir: Path, modes_arg: int | None, basis_arg: str | None):
    candidates = []
    if basis_arg: candidates.append(Path(basis_arg))
    candidates += [
        run_dir / "pod_basis.npz",
        REPO / "outputs" / "pod_basis" / "pod_basis_256.npz",
        REPO / "outputs" / "pod_basis" / "pod_basis_128.npz",
        REPO / "outputs" / "pod_basis" / "pod_basis_96.npz",
        REPO / "outputs" / "pod_basis" / "pod_basis_64.npz",
    ]
    for p in candidates:
        try:
            if not p.exists(): continue
            B = _load_basis_matrix(p)
            cols = int(B.shape[1])
            modes = cols if modes_arg is None else min(int(modes_arg), cols)
            if modes_arg is not None and modes_arg > cols:
                print(f"[WARN] requested --pod_modes {modes_arg} > basis cols {cols}; clamping to {cols}.")
            return str(p), modes
        except Exception:
            continue
    raise FileNotFoundError("No usable POD basis found. Use --pod_basis or place pod_basis_*.npz in outputs/pod_basis/")

# ----- build/load models -----
def build_model(kind, in_ch, out_ch, H, W, device, run_dir: Path, sd: dict,
                pod_modes: int | None, pod_basis: str | None):
    if kind == "fno":
        return FNO2D(in_ch, out_ch, (16,16), layers=4, hidden_channels=64).to(device)
    if kind == "unet":
        base = _infer_unet_base(sd)
        return UNet2D(in_ch, out_ch, base_channels=base, depth=4, dropout=0.0).to(device)
    if kind == "pod":
        basis_path, modes = _resolve_pod_basis(run_dir, pod_modes, pod_basis)
        return PODRegressor(in_channels=in_ch, H=H, W=W, modes=modes, basis_path=basis_path).to(device)
    raise ValueError(f"unknown kind {kind}")

def load_from_run(kind, run_dir: Path, in_ch, out_ch, H, W, device,
                  pod_modes: int | None, pod_basis: str | None):
    ckpt = torch.load(run_dir / "ckpt_best.pt", map_location=device)
    sd = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt.get("model") or {}
    model = build_model(kind, in_ch, out_ch, H, W, device, run_dir, sd, pod_modes, pod_basis)
    miss, unexp = model.load_state_dict(sd, strict=False)
    if miss or unexp:
        print(f"[WARN] load_state_dict: missing={miss} unexpected={unexp}")
    return model

# ----- data & eval -----
def make_loaders(base_cfg, ds_cfg, bs=4):
    cfg = {}; cfg.update(base_cfg); cfg.update(ds_cfg)
    ds_tr = AirfransDataset(cfg, cfg, split="train")
    try:
        ds_va = AirfransDataset(cfg, cfg, split="val")
    except Exception:
        ds_va = AirfransDataset(cfg, cfg, split="test")
    return (DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=0),
            DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=0))

@torch.no_grad()
def eval_nrmse(model, dl, device, mean, std, crop):
    model.eval()
    sse = torch.zeros(3, device=device)
    npx = torch.zeros(3, device=device)
    y_min = torch.full((3,), float("inf"), device=device)
    y_max = torch.full((3,), float("-inf"), device=device)
    for b in dl:
        x = b["inputs"].to(device).float()
        y = b["targets"].to(device).float()
        p = inv_targets(model(x), mean.to(device), std.to(device))
        if crop > 0:
            y = crop2d(y, crop); p = crop2d(p, crop)
        diff2 = (p - y)**2
        sse += diff2.sum(dim=(0,2,3))
        npx += torch.tensor([y[:,0].numel(), y[:,1].numel(), y[:,2].numel()], device=device)
        y_min = torch.minimum(y_min, y.amin(dim=(0,2,3)))
        y_max = torch.maximum(y_max, y.amax(dim=(0,2,3)))
    rmse = torch.sqrt(sse / npx.clamp_min(1))
    rng = (y_max - y_min).clamp_min(1e-6)
    return rmse / rng

def compute_loss(y, p, w, ew):
    yu, yv, yp = y[:,0:1], y[:,1:2], y[:,2:3]
    pu, pv, pp = p[:,0:1], p[:,1:2], p[:,2:3]
    loss = (
        w["wu"] * wmean(F.l1_loss(pu, yu, reduction="none"), ew) +
        w["wv"] * wmean(F.l1_loss(pv, yv, reduction="none"), ew) +
        w["wp"] * wmean(F.l1_loss(pp, yp, reduction="none"), ew)
    )
    if w["w_grad"] > 0:
        loss += w["w_grad"] * (
            wmean(F.l1_loss(cdx(pu), cdx(yu), reduction="none"), ew) +
            wmean(F.l1_loss(cdy(pu), cdy(yu), reduction="none"), ew) +
            wmean(F.l1_loss(cdx(pv), cdx(yv), reduction="none"), ew) +
            wmean(F.l1_loss(cdy(pv), cdy(yv), reduction="none"), ew)
        )
    if w["w_vort"] > 0:
        loss += w["w_vort"] * wmean((vorticity(pu, pv) - vorticity(yu, yv))**2, ew)
    if w["w_div"] > 0:
        loss += w["w_div"] * wmean((divergence(pu, pv))**2, ew)
    return loss

# ==================== main ====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_cfg", required=True)
    ap.add_argument("--dataset_cfg", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--kind", required=True, choices=["fno","unet","pod"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min_lr", type=float, default=5e-5)
    ap.add_argument("--sched", choices=["none","cosine"], default="cosine")
    ap.add_argument("--crop", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    # POD
    ap.add_argument("--pod_modes", type=int, default=None)
    ap.add_argument("--pod_basis", type=str, default=None)
    # loss weights
    ap.add_argument("--wu", type=float, default=6.0)
    ap.add_argument("--wv", type=float, default=1.0)
    ap.add_argument("--wp", type=float, default=0.5)
    ap.add_argument("--w_grad", type=float, default=0.30)
    ap.add_argument("--w_vort", type=float, default=0.45)
    ap.add_argument("--w_div", type=float, default=0.00)
    # edge focus
    ap.add_argument("--edge_focus", action="store_true")
    ap.add_argument("--edge_alpha", type=float, default=4.0)
    ap.add_argument("--edge_gain",  type=float, default=1.5)
    args = ap.parse_args()

    import yaml
    base = yaml.safe_load(open(args.base_cfg))
    dcfg = yaml.safe_load(open(args.dataset_cfg))
    dl_tr, dl_va = make_loaders(base, dcfg, bs=4)

    b0 = next(iter(dl_tr))
    Cin, H, W = int(b0["inputs"].shape[1]), int(b0["inputs"].shape[-2]), int(b0["inputs"].shape[-1])
    device = torch.device(args.device)
    run_dir = Path(args.run)

    model = load_from_run(args.kind, run_dir, Cin, 3, H, W, device,
                          pod_modes=args.pod_modes, pod_basis=args.pod_basis)
    mean, std = load_out_norm(run_dir)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sch = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.min_lr)
           if args.sched == "cosine" else None)

    w = dict(wu=args.wu, wv=args.wv, wp=args.wp, w_grad=args.w_grad, w_vort=args.w_vort, w_div=args.w_div)

    best = math.inf
    best_path = run_dir / "ckpt_finetune_best.pt"

    for e in range(1, args.epochs + 1):
        model.train()
        for b in dl_tr:
            x = b["inputs"].to(device).float()        # NO pre-crop
            y = b["targets"].to(device).float()       # NO pre-crop
            p  = inv_targets(model(x), mean.to(device), std.to(device))  # full size
            # crop AFTER forward so POD matches UNet/FNO
            if args.crop > 0:
                y_l = crop2d(y, args.crop)
                p_l = crop2d(p, args.crop)
            else:
                y_l, p_l = y, p
            ew = edge_weight(y_l[:,0:1], args.edge_alpha, args.edge_gain) if args.edge_focus else None
            loss = compute_loss(y_l, p_l, w, ew)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            nrmse = eval_nrmse(model, dl_va, device, mean, std, args.crop)
        u, v, p = [float(nrmse[i].cpu()) for i in range(3)]
        sel = u + v
        if sch is not None: sch.step()
        print(f"[epoch {e:02d}] lr={opt.param_groups[0]['lr']:.2e} "
              f"val NRMSE(u)={u:.4f}  v={v:.4f}  p={p:.4f}  sum_uv={sel:.4f}  best={best:.4f}")
        if sel < best - 1e-9:
            best = sel
            torch.save({"model_state": model.state_dict()}, best_path)

    print(f"[OK] wrote finetuned checkpoint to {best_path}")
    print("Tip: cp -f <best> <run>/ckpt_best.pt to use it in your eval scripts.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
