import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STATS_ROOT = "stats"
OUT_DIR = "plots"

os.makedirs(OUT_DIR, exist_ok=True)

ARCH_ORDER = ["Sequential", "Parallel", "Gamma", "SparTA"]

records = []

# ------------------------------------------------
# Load results
# ------------------------------------------------

for model in sorted(os.listdir(STATS_ROOT)):

    csv_path = os.path.join(STATS_ROOT, model, "results.csv")

    if not os.path.exists(csv_path):
        continue

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        records.append(
            {
                "model": model,
                "arch": row["Architecture"],
                "ticks": row["simTicks"],
                "energy": row.get("Energy"),
                "edp": row.get("EDP"),
                "power": row.get("runtime_dynamic"),
            }
        )

data = pd.DataFrame(records)

if data.empty:
    raise RuntimeError("No experiment results found")

# ------------------------------------------------
# Pivot table
# ------------------------------------------------

pivot = data.pivot(index="model", columns="arch", values="ticks")

pivot = pivot.reindex(columns=[a for a in ARCH_ORDER if a in pivot.columns])

# ------------------------------------------------
# Runtime plot (log scale)
# ------------------------------------------------

plt.figure(figsize=(12, 6))

pivot.plot(kind="bar", ax=plt.gca())

plt.yscale("log")
plt.ylabel("simTicks (log scale)")
plt.title("Runtime Comparison")
plt.xticks(rotation=60)

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/runtime_bar.png")
plt.close()

print("Saved runtime_bar.png")

# ------------------------------------------------
# Speedup vs Sequential
# ------------------------------------------------

if "Sequential" in pivot.columns:

    speedup = pivot.apply(lambda col: pivot["Sequential"] / col)

    speedup = speedup.drop(columns=["Sequential"], errors="ignore")

    plt.figure(figsize=(12, 6))

    speedup.plot(kind="bar", ax=plt.gca())

    plt.yscale("log")
    plt.ylabel("Speedup vs Sequential")
    plt.title("Architecture Speedup")
    plt.xticks(rotation=60)

    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/speedup_bar.png")
    plt.close()

    print("Saved speedup_bar.png")

# ------------------------------------------------
# Normalized runtime
# ------------------------------------------------

if "Sequential" in pivot.columns:

    norm = pivot.div(pivot["Sequential"], axis=0)

    plt.figure(figsize=(12, 6))

    norm.plot(kind="bar", ax=plt.gca())

    plt.yscale("log")
    plt.ylabel("Normalized Runtime (Sequential = 1)")
    plt.title("Normalized Runtime")
    plt.xticks(rotation=60)

    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/normalized_runtime.png")
    plt.close()

    print("Saved normalized_runtime.png")

# ------------------------------------------------
# Geometric mean speedup
# ------------------------------------------------

if "Sequential" in pivot.columns:

    speedup = pivot.apply(lambda col: pivot["Sequential"] / col)
    speedup = speedup.drop(columns=["Sequential"], errors="ignore")

    gmean = np.exp(np.log(speedup).mean())

    plt.figure(figsize=(8, 5))

    gmean.plot(kind="bar")

    plt.ylabel("Geometric Mean Speedup")
    plt.title("Overall Architecture Speedup")

    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/geomean_speedup.png")
    plt.close()

    print("Saved geomean_speedup.png")

# ------------------------------------------------
# Energy
# ------------------------------------------------

pivot_energy = data.pivot(index="model", columns="arch", values="energy")
pivot_energy = pivot_energy.reindex(columns=ARCH_ORDER)

pivot_energy.plot(kind="bar", figsize=(12, 6))
plt.yscale("log")
plt.ylabel("Energy (J)")
plt.title("Energy Consumption")
plt.xticks(rotation=60)
plt.tight_layout()

plt.savefig(f"{OUT_DIR}/energy_bar.png")
plt.close()

print("Saved energy_bar.png")

# ------------------------------------------------
# Normalized Energy
# ------------------------------------------------

norm_energy = pivot_energy.div(pivot_energy["Sequential"], axis=0)

norm_energy.plot(kind="bar", figsize=(12, 6))
plt.ylabel("Normalized Energy")
plt.title("Energy Normalized to Sequential")
plt.xticks(rotation=60)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/energy_norm.png")
plt.close()

print("Saved energy_norm.png")

# ------------------------------------------------
# EDP
# ------------------------------------------------

pivot_edp = data.pivot(index="model", columns="arch", values="edp")
pivot_edp = pivot_edp.reindex(columns=ARCH_ORDER)

pivot_edp.plot(kind="bar", figsize=(12, 6))
plt.yscale("log")
plt.ylabel("EDP")
plt.title("Energy-Delay Product")
plt.xticks(rotation=60)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/edp_bar.png")
plt.close()

print("Saved edp_bar.png")

print("\nAll plots saved in:", OUT_DIR)
