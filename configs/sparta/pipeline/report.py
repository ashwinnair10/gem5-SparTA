import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ARCH_ORDER = ["Sequential", "Parallel", "Gamma", "SpMard", "SparTA"]


def build_report(results):

    df = pd.DataFrame.from_dict(results, orient="index")

    df.index.name = "Architecture"

    df = df.reindex([a for a in ARCH_ORDER if a in df.index])

    if "Pade" in df.index:
        df.loc["Pade", "simTicks"] = round(df.loc["Pade", "simTicks"] / 8)

    if "Sequential" in df.index:
        seq_ticks = df.loc["Sequential", "simTicks"]
        df["Speedup"] = seq_ticks / df["simTicks"]

    df["runtime_sec"] = df["simTicks"] / float(os.getenv("CLOCK"))
    df["Energy"] = df["runtime_dynamic"] * df["runtime_sec"]
    df["EDP"] = df["Energy"] * df["runtime_sec"]

    return df.reset_index()


def save_csv(df, outdir, name):

    os.makedirs(outdir, exist_ok=True)

    path = os.path.join(outdir, name)

    df.to_csv(path, index=False)

    print(f"[REPORT] Saved {path}")
