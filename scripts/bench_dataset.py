#!/usr/bin/env python3
import argparse, os, sys, csv
from pathlib import Path
import numpy as np
import torch
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset_airfrans import AirfransDataset
from src.models.unet.unet2d import UNet2D
from src.models.fno.fno2d import FNO2D
from src.models.pod.pod_head import PODRegressor

def load_yaml(p):
    with open(p,"r") as f: return yaml.safe_load(f)

def inv_from_run(run_dir: str):
    zpath = Path(run_dir)/"out_norm.npz"
    if zpath.exists():
        z = np.load(zpath)
        mean, std = z["mean"].astype(np.float32), z["std"].astype(np.float32)
        return lambda x: x*std[:,None,None] + mean[:,None,None]
    return lambda x: x

def build(kind, ckpt_path, in_ch, H, W, device, pod_modes=64, pod_basis=None):
    if kind=="unet":
        m = UNet2D(in_channels=in_ch, out_channels=3, base_channels=32, depth=4, dropout=0.0)
    elif kind=="fno":
        m = FNO2D(in_channels=in_ch, out_channels=3, modes=(16,16), layers=4, hidden_channels=64)
    elif kind=="pod":
        if pod_basis is None or not Path(pod_basis).exists():
            # try defaults
            cand = [
                Path("outputs/pod_basis")/f"pod_basis_{pod_modes}.npz",
                Path("outputs/pod_basis")/"pod_basis_64.npz",
            ]
            for c in cand:
                if c.exists(): pod_basis=str(c); break
        if pod_basis is None or not Path(pod_basis).exists():
            raise FileNotFoundError("POD basis not found; pass --pod_basis or place outputs/pod_basis/pod_basis_*.npz")
        m = PODRegressor(in_channels=in_ch, H=H, W=W, modes=pod_modes, basis_path=pod_basis)
    else:
        raise ValueError(kind)

    sd = torch.load(str(ckpt_path), map_location="cpu")
    state = sd.get("model_state") or sd.get("state_dict") or sd.get("model") or sd
    if isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]
    m.load_state_dict(state, strict=False)
    return m.to(device).eval()

def mae(a,b,crop=0):
    if crop>0: a=a[:,crop:-crop,crop:-crop]; b=b[:,crop:-crop,crop:-crop]
    return np.mean(np.abs(a-b),axis=(1,2))

def rmse(a,b,crop=0):
    if crop>0: a=a[:,crop:-crop,crop:-crop]; b=b[:,crop:-crop,crop:-crop]
    return np.sqrt(np.mean((a-b)**2,axis=(1,2)))

def nrmse_range(a,b,crop=0):
    r = rmse(a,b,crop)
    span = np.maximum(np.array([b[i].max()-b[i].min() for i in range(3)]), 1e-6)
    return r / span

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_cfg", type=str, required=True)
    ap.add_argument("--dataset_cfg", type=str, required=True)
    ap.add_argument("--unet_run", type=str, required=True)
    ap.add_argument("--fno_run",  type=str, required=True)
    ap.add_argument("--pod_run",  type=str, required=True)
    ap.add_argument("--pod_basis", type=str, default=None)
    ap.add_argument("--pod_modes", type=int, default=64)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--crop", type=int, default=8)
    args = ap.parse_args()

    base = load_yaml(args.base_cfg); dcfg = load_yaml(args.dataset_cfg)
    cfg={}; cfg.update(base); cfg.update(dcfg)

    ds = AirfransDataset(cfg,cfg,"test")
    Cin,H,W = ds[0]["inputs"].shape[-3], ds[0]["inputs"].shape[-2], ds[0]["inputs"].shape[-1]
    dev = torch.device(args.device)

    models = {
        "UNet": build("unet", Path(args.unet_run)/"ckpt_best.pt", Cin,H,W,dev),
        "FNO" : build("fno" , Path(args.fno_run )/"ckpt_best.pt", Cin,H,W,dev),
        "POD" : build("pod" , Path(args.pod_run )/"ckpt_best.pt", Cin,H,W,dev,
                      pod_modes=args.pod_modes, pod_basis=args.pod_basis),
    }
    invs = {
        "UNet": inv_from_run(args.unet_run),
        "FNO" : inv_from_run(args.fno_run),
        "POD" : inv_from_run(args.pod_run),
    }

    out_path = Path(args.out_csv); out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id","model","crop",
                    "u_mae","v_mae","p_mae",
                    "u_rmse","v_rmse","p_rmse",
                    "u_nrmse","v_nrmse","p_nrmse"])
        for i in range(len(ds)):
            b = ds[i]
            case = b.get("meta",{}).get("case_id", f"case{i:03d}")
            x = b["inputs"].to(dev); gt = b["targets"].numpy()
            with torch.no_grad():
                for name, m in models.items():
                    pr = m(x[None]).cpu().numpy()[0]
                    pr = invs[name](pr)  # inverse predictions only
                    w.writerow([
                        case, name, args.crop,
                        *[float(v) for v in mae(pr,gt,args.crop)],
                        *[float(v) for v in rmse(pr,gt,args.crop)],
                        *[float(v) for v in nrmse_range(pr,gt,args.crop)],
                    ])
    print(f"[OK] wrote {args.out_csv}")

if __name__=="__main__":
    main()
