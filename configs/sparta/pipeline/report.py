# configs/sparta/pipeline/report.py

import os

import pandas as pd

ARCH_ORDER = [
    "Sequential",
    "Parallel",
    "Gamma",
    "SparTA",
]


def build_report(results):

    df = pd.DataFrame.from_dict(results, orient="index")

    df.index.name = "Architecture"

    # enforce architecture order if present
    df = df.reindex([a for a in ARCH_ORDER if a in df.index])

    # compute speedup vs sequential
    if "Sequential" in df.index:
        seq_ticks = df.loc["Sequential", "simTicks"]
        df["Speedup"] = seq_ticks / df["simTicks"]

    return df.reset_index()


def save_csv(df, outdir, name):

    os.makedirs(outdir, exist_ok=True)

    path = os.path.join(outdir, name)

    df.to_csv(path, index=False)

    print(f"[REPORT] Saved {path}")
