import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STATS_ROOT = "stats"
OUT_DIR = "plots"

os.makedirs(OUT_DIR, exist_ok=True)

ARCH_ORDER = ["Sequential", "Parallel", "Gamma", "SpMard", "SparTA"]

records = []

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

data = data[data["model"] != "default"]
data = data[~data["model"].str.startswith("prajjwal1_bert-tiny")]

MODEL_MAP = {
    "allenai_longformer-base-4096": "longformer",
    "google_bigbird-roberta-base": "bigbird",
    "bert-base-uncased": "bert-base",
    "prajjwal1_bert-tiny": "bert-tiny",
    "gpt2": "gpt2",
    "google_vit-base-patch16-224": "vit",
}


def map_model_name(full_name):
    if full_name == "default":
        return
    parts = full_name.split("_")

    sparsity = parts[-1]
    dataset = parts[-2]
    prefix = "_".join(parts[:-2])

    for key in MODEL_MAP:
        if prefix.startswith(key):
            model = MODEL_MAP[key]
            break
    else:
        model = prefix
    return f"{model}_{dataset}_{sparsity}"


data["model_name"] = data["model"].apply(map_model_name)


pivot = data.pivot(index="model_name", columns="arch", values="ticks")

pivot = pivot.reindex(columns=[a for a in ARCH_ORDER if a in pivot.columns])

plt.figure(figsize=(12, 6))

pivot.plot(kind="bar", ax=plt.gca())

plt.yscale("log")
plt.ylabel("simTicks (log scale)")
plt.title("Runtime Comparison")
plt.xticks(rotation=90)

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/runtime_bar.png")
plt.close()

print("Saved runtime_bar.png")

if "Sequential" in pivot.columns:

    speedup = pivot.apply(lambda col: pivot["Sequential"] / col)

    speedup = speedup.drop(columns=["Sequential"], errors="ignore")

    plt.figure(figsize=(12, 6))

    speedup.plot(kind="bar", ax=plt.gca())

    plt.yscale("log")
    plt.ylabel("Speedup vs Sequential")
    plt.title("Architecture Speedup")
    plt.xticks(rotation=90)

    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/speedup_bar.png")
    plt.close()

    print("Saved speedup_bar.png")

if "Sequential" in pivot.columns:

    norm = pivot.div(pivot["Sequential"], axis=0)

    plt.figure(figsize=(12, 6))

    norm.plot(kind="bar", ax=plt.gca())

    plt.yscale("log")
    plt.ylabel("Normalized Runtime (Sequential = 1)")
    plt.title("Normalized Runtime")
    plt.xticks(rotation=90)

    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/normalized_runtime.png")
    plt.close()

    print("Saved normalized_runtime.png")

if "Sequential" in pivot.columns:

    speedup = pivot.apply(lambda col: pivot["SpMard"] / col)
    speedup = speedup.drop(columns=["SpMard"], errors="ignore")

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

pivot_energy = data.pivot(index="model_name", columns="arch", values="energy")
pivot_energy = pivot_energy.reindex(columns=ARCH_ORDER)

pivot_energy.plot(kind="bar", figsize=(12, 6))
plt.yscale("log")
plt.ylabel("Energy (J)")
plt.title("Energy Consumption")
plt.xticks(rotation=90)
plt.tight_layout()

plt.savefig(f"{OUT_DIR}/energy_bar.png")
plt.close()

print("Saved energy_bar.png")

norm_energy = pivot_energy.div(pivot_energy["Sequential"], axis=0)

norm_energy.plot(kind="bar", figsize=(12, 6))
plt.ylabel("Normalized Energy")
plt.title("Energy Normalized to Sequential")
plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/energy_norm.png")
plt.close()

print("Saved energy_norm.png")

pivot_edp = data.pivot(index="model_name", columns="arch", values="edp")
pivot_edp = pivot_edp.reindex(columns=ARCH_ORDER)

pivot_edp.plot(kind="bar", figsize=(12, 6))
plt.yscale("log")
plt.ylabel("EDP")
plt.title("Energy-Delay Product")
plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/edp_bar.png")
plt.close()

print("Saved edp_bar.png")

print("\nAll plots saved in:", OUT_DIR)
print("\n" + "=" * 50)
print("SPARTA PERFORMANCE & SAVINGS SUMMARY")
print("=" * 50)

if "SparTA" in pivot.columns:
    print("\n[Speedup] SparTA is X times faster than:")
    for arch in ARCH_ORDER:
        if arch in pivot.columns and arch != "SparTA":
            sparta_speedup = (pivot[arch] / pivot["SparTA"]).mean()
            print(f"  vs {arch:12}: {sparta_speedup:.2f}x")

if "SparTA" in pivot_energy.columns:
    print("\n[Energy] SparTA uses X% of the energy of:")
    # Savings = (SparTA_Energy / Other_Energy) * 100
    for arch in ARCH_ORDER:
        if arch in pivot_energy.columns and arch != "SparTA":
            energy_pct = (
                pivot_energy["SparTA"] / pivot_energy[arch]
            ).mean() * 100
            savings = 100 - energy_pct
            print(f"  vs {arch:12}: {energy_pct:.1f}% (saves {savings:.1f}%)")

if "SparTA" in pivot_edp.columns:
    print("\n[EDP] SparTA improvement (reduction) wrt:")
    for arch in ARCH_ORDER:
        if arch in pivot_edp.columns and arch != "SparTA":
            edp_reduction = (pivot_edp[arch] / pivot_edp["SparTA"]).mean()
            print(f"  vs {arch:12}: {edp_reduction:.2f}x better")

print("=" * 50)
