#!/usr/bin/env python3
import argparse, yaml, numpy as np
from pathlib import Path
import pandas as pd
import torch
import matplotlib.pyplot as plt

# --- project imports (unchanged) ---
from src.data.dataset_airfrans import AirfransDataset
from src.models.fno.fno2d import FNO2D
from src.models.unet.unet2d import UNet2D
from src.models.pod.pod_head import PODRegressor

# -------- helpers ----------
def load_yaml(p):
    with open(p,"r") as f: 
        return yaml.safe_load(f)

def inv_from_run(run_dir: Path):
    npz = np.load(run_dir / "out_norm.npz")
    mean = npz["mean"].astype(np.float32); std = npz["std"].astype(np.float32)
    return lambda arr: arr * std[:,None,None] + mean[:,None,None]

def build_model(kind, in_ch, out_ch, H, W, device, pod_basis=None, pod_modes=64):
    if kind == "fno":
        m = FNO2D(in_ch, out_ch, modes=(16,16), layers=4, hidden_channels=64)
    elif kind == "unet":
        m = UNet2D(in_ch, out_ch, base_channels=32, depth=4, dropout=0.0)
    elif kind == "pod":
        assert pod_basis is not None, "Pass --pod_basis for POD"
        m = PODRegressor(in_channels=in_ch, H=H, W=W, modes=pod_modes, basis_path=pod_basis)
    else:
        raise ValueError(kind)
    return m.to(device).eval()

def load_ckpt(model, ckpt_path):
    sd = torch.load(str(ckpt_path), map_location="cpu")
    state = sd.get("model_state", sd)
    model.load_state_dict(state, strict=True)
    return model

def predict(model, x):
    with torch.no_grad():
        pr = model(x).squeeze(0).cpu().numpy()
    return pr

# ----- very simple pressure-only CL/CD from mask edge -----
def surface_indices(mask):
    # mask is (H,W), 1 on body. Extract 1-pixel wide boundary.
    from scipy.ndimage import binary_erosion
    edge = mask.astype(bool) & (~binary_erosion(mask.astype(bool)))
    ys, xs = np.where(edge)
    return ys, xs

def cp_from_p(p, u_inf, rho=1.225, p_inf=0.0):
    q = 0.5 * rho * (u_inf**2 + 1e-8)
    return (p - p_inf) / q

def integrate_forces(cp, mask):
    # crude finite-area integration; assume unit dx=dy and Sref=area(mask)
    ys, xs = surface_indices(mask)
    if len(xs) == 0:
        return 0.0, 0.0
    # normals via image gradients of mask
    gy, gx = np.gradient(mask.astype(np.float32))
    nx = -gx; ny = -gy
    nx = nx[ys, xs]; ny = ny[ys, xs]
    nrm = np.sqrt(nx**2 + ny**2) + 1e-8
    nx /= nrm; ny /= nrm
    # pressure force ~ -cp * n dS ; Sref = area(mask)
    dS = 1.0
    Fx = float(np.sum(-cp[ys, xs] * nx * dS))
    Fy = float(np.sum(-cp[ys, xs] * ny * dS))
    Sref = float(mask.sum() + 1e-8)
    # return coefficients (normalized by Sref)
    return Fx / Sref, Fy / Sref

# --------------- main -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_cfg", type=str, default="configs/base.yaml")
    ap.add_argument("--dataset_cfg", type=str, default="configs/dataset_airfrans.yaml")
    ap.add_argument("--unet_run", type=str, required=True)
    ap.add_argument("--fno_run", type=str, required=True)
    ap.add_argument("--pod_run", type=str, required=True)
    ap.add_argument("--pod_basis", type=str, default=None)
    ap.add_argument("--pod_modes", type=int, default=64)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max_cases", type=int, default=50)
    ap.add_argument("--outdir", type=str, default="Final_outputs/aero")
    args = ap.parse_args()

    base = load_yaml(args.base_cfg); dcfg = load_yaml(args.dataset_cfg)
    cfg = {}; cfg.update(base); cfg.update(dcfg)

    ds = AirfransDataset(cfg, cfg, split="test")
    b0 = ds[0]
    Cin, H, W = int(b0["inputs"].shape[0]), int(b0["targets"].shape[-2]), int(b0["targets"].shape[-1])
    device = torch.device(args.device)

    runs = {
        "GT"  : None,  # handled separately
        "UNet": Path(args.unet_run),
        "FNO" : Path(args.fno_run),
        "POD" : Path(args.pod_run),
    }

    models = {}
    invf   = {}
    for name, rdir in runs.items():
        if name == "GT": 
            continue
        if name == "POD":
            m = build_model("pod", Cin, 3, H, W, device, pod_basis=args.pod_basis, pod_modes=args.pod_modes)
        elif name == "FNO":
            m = build_model("fno", Cin, 3, H, W, device)
        else:
            m = build_model("unet", Cin, 3, H, W, device)
        load_ckpt(m, rdir / "ckpt_best.pt")
        models[name] = m
        invf[name]   = inv_from_run(rdir)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    rows = []

    for i in range(min(len(ds), args.max_cases)):
        b = ds[i]
        x  = b["inputs"][None].to(device)   # (1,4,H,W)
        gt = b["targets"].numpy()           # (3,H,W)
        case_id = b.get("meta",{}).get("case_id", f"case{i:03d}")

        # fixed channel indices: [sdf, mask, aoa, u_inf]
        mask = b["inputs"][1].numpy()
        aoa   = float(b["inputs"][2].mean().item())
        u_inf = float(b["inputs"][3].mean().item())

        preds = {"GT": gt}
        for name, m in models.items():
            pr = predict(m, x)
            pr = invf[name](pr)
            preds[name] = pr

        # CL/CD from pressure-only (crude)
        cp_gt  = cp_from_p(preds["GT"][2], u_inf)
        Fx_gt, Fy_gt = integrate_forces(cp_gt, mask)
        cl_gt, cd_gt = Fy_gt, Fx_gt

        rec = {"case": case_id, "aoa": aoa, "u_inf": u_inf,
               "CL_GT": cl_gt, "CD_GT": cd_gt}
        for name in ["UNet","FNO","POD"]:
            cp = cp_from_p(preds[name][2], u_inf)
            Fx, Fy = integrate_forces(cp, mask)
            rec[f"CL_{name}"] = Fy
            rec[f"CD_{name}"] = Fx
        rows.append(rec)

        # save Cp line (mid-height) for a quick visual
        ymid = H//2
        plt.figure(figsize=(6,3))
        xcoord = np.arange(W)
        for name in ["GT","UNet","FNO","POD"]:
            plt.plot(xcoord, cp_from_p(preds[name][2], u_inf)[ymid], label=name)
        plt.xlabel("x-index"); plt.ylabel("Cp"); plt.title(f"{case_id} – midline Cp")
        plt.legend(); plt.tight_layout()
        plt.savefig(outdir / f"{case_id}_cp.png", dpi=140)
        plt.close()

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "cl_cd.csv", index=False)

    # quick scatter plots
    for coef in ["CL","CD"]:
        plt.figure(figsize=(5,4))
        plt.scatter(df[f"{coef}_GT"], df[f"{coef}_UNet"], label="UNet", s=10)
        plt.scatter(df[f"{coef}_GT"], df[f"{coef}_FNO"],  label="FNO",  s=10)
        plt.scatter(df[f"{coef}_GT"], df[f"{coef}_POD"],  label="POD",  s=10)
        lo = float(min(df[f"{coef}_GT"].min(), df.filter(like=coef+"_").drop(columns=[f"{coef}_GT"]).min().min()))
        hi = float(max(df[f"{coef}_GT"].max(), df.filter(like=coef+"_").drop(columns=[f"{coef}_GT"]).max().max()))
        plt.plot([lo,hi],[lo,hi],"k--",linewidth=1)
        plt.xlabel(f"{coef} GT"); plt.ylabel(f"{coef} pred")
        plt.title(f"{coef} scatter")
        plt.legend(); plt.tight_layout()
        plt.savefig(outdir / f"scatter_{coef.lower()}.png", dpi=140)
        plt.close()

    print(f"[OK] wrote CL/CD and plots to {outdir}")

if __name__ == "__main__":
    main()
