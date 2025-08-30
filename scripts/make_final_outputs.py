#!/usr/bin/env python3
import argparse, os, sys
from pathlib import Path
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset_airfrans import AirfransDataset
from src.models.unet.unet2d import UNet2D
from src.models.fno.fno2d import FNO2D
from src.models.pod.pod_head import PODRegressor

UNITS = {"u":"m/s","v":"m/s","p":"Pa","U":"m/s"}

def load_yaml(p):
    with open(p,"r") as f: return yaml.safe_load(f)

def inv_from_run(run_dir: str):
    zpath = Path(run_dir)/"out_norm.npz"
    if zpath.exists():
        z = np.load(zpath)
        mean, std = z["mean"].astype(np.float32), z["std"].astype(np.float32)
        return lambda x: x*std[:,None,None] + mean[:,None,None]
    return lambda x: x

def build(kind, ckpt_path, in_ch, H, W, device="cpu", pod_modes=64, pod_basis=None):
    if kind=="unet":
        m = UNet2D(in_channels=in_ch, out_channels=3, base_channels=32, depth=4, dropout=0.0)
    elif kind=="fno":
        m = FNO2D(in_channels=in_ch, out_channels=3, modes=(16,16), layers=4, hidden_channels=64)
    elif kind=="pod":
        if pod_basis is None or not Path(pod_basis).exists():
            cand = [
                Path("outputs/pod_basis")/f"pod_basis_{pod_modes}.npz",
                Path("outputs/pod_basis")/"pod_basis_64.npz",
            ]
            for c in cand:
                if c.exists(): pod_basis=str(c); break
        if pod_basis is None or not Path(pod_basis).exists():
            raise FileNotFoundError("POD basis not found; pass --pod_basis")
        m = PODRegressor(in_channels=in_ch, H=H, W=W, modes=pod_modes, basis_path=pod_basis)
    else:
        raise ValueError(kind)

    sd = torch.load(str(ckpt_path), map_location="cpu")
    state = sd.get("model_state") or sd.get("state_dict") or sd.get("model") or sd
    if isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]
    m.load_state_dict(state, strict=False)
    return m.to(device).eval()

def add_cbar(ax, im, label):
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="3%", pad=0.02)
    cb = ax.figure.colorbar(im, cax=cax)
    cb.ax.set_ylabel(label)

def fields_and_U(arr3):
    u,v,p = arr3
    U = np.sqrt(u*u + v*v)
    return {"u":u,"v":v,"p":p,"U":U}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base_cfg", type=str, required=True)
    ap.add_argument("--dataset_cfg", type=str, required=True)
    ap.add_argument("--unet_run", type=str, required=True)
    ap.add_argument("--fno_run",  type=str, required=True)
    ap.add_argument("--pod_run",  type=str, required=True)
    ap.add_argument("--pod_basis", type=str, default="outputs/pod_basis/pod_basis_64.npz")
    ap.add_argument("--pod_modes", type=int, default=64)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max_cases", type=int, default=8)
    ap.add_argument("--global_minmax", action="store_true")
    ap.add_argument("--outdir", type=str, default="Final_outputs")
    ap.add_argument("--cmap_field", type=str, default="turbo")
    ap.add_argument("--cmap_error", type=str, default="seismic")
    ap.add_argument("--symmetric_error", action="store_true")
    args=ap.parse_args()

    base = load_yaml(args.base_cfg); dcfg = load_yaml(args.dataset_cfg)
    cfg={}; cfg.update(base); cfg.update(dcfg)

    ds = AirfransDataset(cfg,cfg,"test")
    b0 = ds[0]
    Cin,H,W = b0["inputs"].shape[-3], b0["inputs"].shape[-2], b0["inputs"].shape[-1]
    dev = torch.device(args.device)

    runs = {"UNet":args.unet_run, "FNO":args.fno_run, "POD":args.pod_run}
    models = {
      "UNet": build("unet", Path(args.unet_run)/"ckpt_best.pt", Cin,H,W, device=dev),
      "FNO" : build("fno" , Path(args.fno_run )/"ckpt_best.pt", Cin,H,W, device=dev),
      "POD" : build("pod" , Path(args.pod_run )/"ckpt_best.pt", Cin,H,W, device=dev,
                    pod_modes=args.pod_modes, pod_basis=args.pod_basis),
    }
    invs = {k: inv_from_run(runs[k]) for k in runs}

    out_root = Path(args.outdir)
    d_mont = out_root/"contours_montage"; d_sep = out_root/"contours_separate"
    d_metrics = out_root/"metrics"; d_plots = out_root/"plots"
    for d in [d_mont,d_sep,d_metrics,d_plots]: d.mkdir(parents=True, exist_ok=True)

    gmin=gmax=None
    if args.global_minmax:
        mins=[]; maxs=[]
        for i in range(min(args.max_cases, len(ds))):
            y = ds[i]["targets"].numpy()
            mins.append(y.min(axis=(1,2))); maxs.append(y.max(axis=(1,2)))
        gmin = np.stack(mins).min(axis=0); gmax = np.stack(maxs).max(axis=0)

    import csv
    csv_path = d_metrics/"per_case_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id","model","u_MAE","v_MAE","p_MAE","U_MAE",
                               "u_RMSE","v_RMSE","p_RMSE","U_RMSE"])

    for i in range(min(args.max_cases,len(ds))):
        b = ds[i]; case = b.get("meta",{}).get("case_id", f"case{i:03d}")
        x = b["inputs"].to(dev); gt = b["targets"].numpy()
        gt_fields = fields_and_U(gt)

        pr = {}
        with torch.no_grad():
            for name,m in models.items():
                p3 = m(x[None]).cpu().numpy()[0]
                p3 = invs[name](p3)
                pr[name] = fields_and_U(p3)

        # montage u,v,p
        names = ["u","v","p"]
        fig,axes = plt.subplots(3,4,figsize=(12,7))
        for r,ch in enumerate(names):
            vmin = gmin[r] if gmin is not None else min([gt_fields[ch].min(), *(pr[n][ch].min() for n in pr)])
            vmax = gmax[r] if gmax is not None else max([gt_fields[ch].max(), *(pr[n][ch].max() for n in pr)])
            mats = {"GT":gt_fields[ch], **{n:pr[n][ch] for n in ["UNet","FNO","POD"]}}
            for c,key in enumerate(["GT","UNet","FNO","POD"]):
                ax=axes[r,c]; im=ax.imshow(mats[key], origin="lower", cmap=args.cmap_field, vmin=vmin, vmax=vmax)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{ch} {key}")
                add_cbar(ax, im, f"{ch} [{UNITS[ch]}]")
        fig.suptitle(f"{case}", y=0.99)
        fig.savefig(d_mont/f"{case}_fields.png"); plt.close(fig)

        # error panels
        fig2,axes2 = plt.subplots(3,3,figsize=(10,7))
        for r,ch in enumerate(names):
            if args.symmetric_error:
                E = max(*(np.nanmax(np.abs(pr[n][ch]-gt_fields[ch])) for n in pr))
                vmin,vmax = -E,E
            else:
                vmin=vmax=None,None
            for c,model in enumerate(["UNet","FNO","POD"]):
                err = pr[model][ch] - gt_fields[ch]
                ax=axes2[r,c]; im=ax.imshow(err, origin="lower", cmap=args.cmap_error, vmin=vmin, vmax=vmax)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{ch} error {model}")
                add_cbar(ax, im, f"{ch} error [{UNITS[ch]}]")
        fig2.suptitle(f"{case}", y=0.99)
        fig2.savefig(d_mont/f"{case}_errors.png"); plt.close(fig2)

        # separate per-field per-model PNGs (u,v,p,U)
        for ch in ["u","v","p","U"]:
            vmin = gt_fields[ch].min(); vmax = gt_fields[ch].max()
            for model in ["GT","UNet","FNO","POD"]:
                arr = gt_fields[ch] if model=="GT" else pr[model][ch]
                fig,ax = plt.subplots(figsize=(4.5,3.6))
                im = ax.imshow(arr, origin="lower", cmap=args.cmap_field, vmin=vmin, vmax=vmax)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{case} – {model} – {ch}")
                add_cbar(ax, im, f"{ch} [{UNITS[ch]}]")
                out_dir = d_sep/ case
                out_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(out_dir/f"{model}_{ch}.png"); plt.close(fig)

        # KPIs
        def _mae(a,b): return float(np.mean(np.abs(a-b)))
        def _rm(a,b):  return float(np.sqrt(np.mean((a-b)**2)))
        for name in ["UNet","FNO","POD"]:
            uM=_mae(pr[name]["u"],gt_fields["u"]); vM=_mae(pr[name]["v"],gt_fields["v"])
            pM=_mae(pr[name]["p"],gt_fields["p"]); UM=_mae(pr[name]["U"],gt_fields["U"])
            uR=_rm (pr[name]["u"],gt_fields["u"]); vR=_rm (pr[name]["v"],gt_fields["v"])
            pR=_rm (pr[name]["p"],gt_fields["p"]); UR=_rm (pr[name]["U"],gt_fields["U"])
            with open(csv_path, "a", newline="") as f2:
                w2 = csv.writer(f2)
                w2.writerow([case,name,uM,vM,pM,UM,uR,vR,pR,UR])

    # summary means
    import pandas as pd
    df = pd.read_csv(csv_path)
    g = df.groupby("model").mean(numeric_only=True)
    g.to_csv(d_metrics/"summary_means.csv")
    with open(d_metrics/"README.txt","w") as f:
        f.write("Per-case metrics in per_case_metrics.csv; means in summary_means.csv\n")
        f.write("Contours (montage + separate) under contours_*/\n")
    print(f"[OK] wrote final outputs under {args.outdir}")

if __name__=="__main__":
    main()
