import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STATS_ROOT = "stats"
OUT_DIR = "plots"

os.makedirs(OUT_DIR, exist_ok=True)

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
            }
        )

data = pd.DataFrame(records)

if data.empty:
    raise RuntimeError("No experiment results found")

# ------------------------------------------------
# Pivot table
# ------------------------------------------------

pivot = data.pivot(index="model", columns="arch", values="ticks")

# ------------------------------------------------
# 1. Runtime bar chart
# ------------------------------------------------

pivot.plot(kind="bar", figsize=(10, 6))

plt.ylabel("simTicks")
plt.title("Runtime Comparison")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(f"{OUT_DIR}/runtime_bar.png")
print("Saved runtime_bar.png")

plt.close()

# ------------------------------------------------
# 2. Speedup vs Sequential
# ------------------------------------------------

if "Sequential" in pivot.columns:

    speedup = pivot.copy()

    for arch in pivot.columns:
        speedup[arch] = pivot["Sequential"] / pivot[arch]

    speedup.drop(columns=["Sequential"], inplace=True)

    speedup.plot(kind="bar", figsize=(10, 6))

    plt.ylabel("Speedup vs Sequential")
    plt.title("Architecture Speedup")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(f"{OUT_DIR}/speedup_bar.png")
    print("Saved speedup_bar.png")

    plt.close()

# ------------------------------------------------
# 3. Normalized runtime
# ------------------------------------------------

norm = pivot.div(pivot["Sequential"], axis=0)

norm.plot(kind="bar", figsize=(10, 6))

plt.ylabel("Normalized Runtime")
plt.title("Normalized Runtime (Sequential = 1)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(f"{OUT_DIR}/normalized_runtime.png")
print("Saved normalized_runtime.png")

plt.close()

# ------------------------------------------------
# 4. Histogram distribution
# ------------------------------------------------

plt.figure(figsize=(8, 6))

for arch in data["arch"].unique():
    subset = data[data["arch"] == arch]
    plt.hist(subset["ticks"], bins=10, alpha=0.6, label=arch)

plt.xlabel("simTicks")
plt.ylabel("Frequency")
plt.title("Tick Distribution")
plt.legend()
plt.tight_layout()

plt.savefig(f"{OUT_DIR}/tick_histogram.png")
print("Saved tick_histogram.png")

plt.close()

# ------------------------------------------------
# 5. Geometric mean speedup
# ------------------------------------------------

if "Sequential" in pivot.columns:

    speedup = pivot.copy()

    for arch in pivot.columns:
        speedup[arch] = pivot["Sequential"] / pivot[arch]

    speedup = speedup.drop(columns=["Sequential"])

    gmean = np.exp(np.log(speedup).mean())

    gmean.plot(kind="bar", figsize=(8, 5))

    plt.ylabel("Geometric Mean Speedup")
    plt.title("Overall Architecture Speedup")
    plt.tight_layout()

    plt.savefig(f"{OUT_DIR}/geomean_speedup.png")
    print("Saved geomean_speedup.png")

    plt.close()

print("\nAll plots saved in:", OUT_DIR)
