#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["UNet", "FNO", "POD"]
CHS = ["u", "v", "p"]

def find_col(df, metric, ch):
    # accept both "u_mae" and "mae_u" (and a few variants)
    candidates = [
        f"{ch}_{metric}", f"{metric}_{ch}",
        f"{ch}{metric}",  f"{metric}{ch}",
        f"{ch.upper()}_{metric}", f"{metric}_{ch.upper()}",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def agg_means(df, cols):
    g = df.groupby("model")[cols].mean()
    # keep only known models, in order
    g = g.reindex(MODELS).dropna(how="all")
    return g

def save_bar(df, metric, out_png, title):
    cols = []
    for ch in CHS:
        c = find_col(df, metric, ch)
        if c:
            cols.append(c)
    if not cols:
        print(f"[WARN] skip {metric}: no matching columns found")
        return
    g = agg_means(df, cols)
    if g.empty:
        print(f"[WARN] skip {metric}: no rows after grouping")
        return
    ax = g.plot(kind="bar", rot=0)
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    # legend labels = channel names
    labels = []
    for c in cols:
        # try to recover channel letter from column name
        for ch in CHS + [c.upper() for c in CHS]:
            if ch in c:
                labels.append(ch.lower())
                break
        else:
            labels.append(c)
    ax.legend(labels, title="channel")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    # If NRMSE appears under a different base name (e.g., "nrmse_range"), alias it
    has_nrmse = any(find_col(df, "nrmse", ch) for ch in CHS)
    if not has_nrmse:
        has_nrmse_range = any(find_col(df, "nrmse_range", ch) for ch in CHS)
        if has_nrmse_range:
            for ch in CHS:
                src = find_col(df, "nrmse_range", ch)
                if src and f"{ch}_nrmse" not in df.columns:
                    df[f"{ch}_nrmse"] = df[src]

    save_bar(df, "mae",  outdir/"mae.png",  "Mean Absolute Error (per channel)")
    save_bar(df, "rmse", outdir/"rmse.png", "RMSE (per channel)")
    save_bar(df, "nrmse", outdir/"nrmse.png", "Normalized RMSE (per channel)")

    print(f"[OK] wrote plots to {outdir}")

if __name__ == "__main__":
    main()
