# configs/sparta/pipeline/report.py
import os

import pandas as pd


def build_report(results):
    df = pd.DataFrame.from_dict(results, orient="index")
    seq = df.loc["Sequential", "simTicks"]
    df["Speedup"] = seq / df["simTicks"]
    df["EDP"] = df["runtime_dynamic"] * df["simTicks"]
    df["EDPNorm_vs_SparTA"] = df["EDP"] / df.loc["SparTA", "EDP"]
    return df


def save_csv(df, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    df.to_csv(path)
    print(f"[REPORT] Saved {path}")
