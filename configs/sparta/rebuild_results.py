import os

import pandas as pd

from configs.sparta.pipeline.mcpat import parse_mcpat
from configs.sparta.pipeline.stats import extract_simticks

STATS_ROOT = "stats"

ARCH_NAME_MAP = {
    "sequential": "Sequential",
    "parallel": "Parallel",
    "gamma": "Gamma",
    "spmard": "SpMard",
    "sparta": "SparTA",
}


def collect_model_results(model_dir):
    results = []

    for arch_dir in os.listdir(model_dir):
        arch_path = os.path.join(model_dir, arch_dir)

        if not os.path.isdir(arch_path):
            continue

        stats_file = os.path.join(arch_path, "stats.txt")
        mcpat_out = os.path.join(arch_path, "mcpat.out")

        if not os.path.exists(stats_file):
            continue

        arch_name = ARCH_NAME_MAP.get(arch_dir.lower(), arch_dir)

        try:
            ticks = extract_simticks(stats_file)
        except:
            ticks = None

        energy_data = {}
        if os.path.exists(mcpat_out):
            try:
                energy_data = parse_mcpat(mcpat_out)
            except:
                energy_data = {}

        results.append(
            {"Architecture": arch_name, "simTicks": ticks, **energy_data}
        )

    return results


def rebuild_all():
    for model in os.listdir(STATS_ROOT):
        model_dir = os.path.join(STATS_ROOT, model)

        if not os.path.isdir(model_dir):
            continue

        csv_path = os.path.join(model_dir, "results.csv")

        if not os.path.exists(csv_path):
            continue

        results = collect_model_results(model_dir)

        if not results:
            continue

        df = pd.DataFrame(results)

        CLOCK = float(os.getenv("CLOCK", 1e12))

        df["runtime_sec"] = df["simTicks"] / CLOCK
        df["Energy"] = df["runtime_dynamic"] * df["runtime_sec"]
        df["EDP"] = df["Energy"] * df["runtime_sec"]

        archs = set(df["Architecture"])

        if len(archs) < 5:
            out_path = os.path.join(model_dir, "results.csv")
            if os.path.exists(out_path):
                os.remove(out_path)
                print(f"[REMOVED - incomplete (<5 archs)] {out_path}")
            continue

        # Sort nicely
        order = ["Sequential", "Parallel", "Gamma", "SpMard", "SparTA"]
        df["Architecture"] = pd.Categorical(df["Architecture"], order)
        df = df.sort_values("Architecture")

        df.to_csv(csv_path, index=False)

        print(f"[✔] Rebuilt: {csv_path}")


if __name__ == "__main__":
    rebuild_all()
