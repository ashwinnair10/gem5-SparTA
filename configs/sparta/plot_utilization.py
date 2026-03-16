import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STATS_ROOT = "stats"
OUT_DIR = "plots"

os.makedirs(OUT_DIR, exist_ok=True)

records = []

for model in sorted(os.listdir(STATS_ROOT)):

    model_dir = os.path.join(STATS_ROOT, model)

    if not os.path.isdir(model_dir):
        continue

    for arch in os.listdir(model_dir):

        stats_path = os.path.join(model_dir, arch, "stats.txt")

        if not os.path.exists(stats_path):
            continue

        mul_active = {}
        mul_idle = {}

        acc_active = {}
        acc_idle = {}

        with open(stats_path) as f:
            for line in f:

                m = re.search(r"mul_units(\d+)\.activeCycles\s+(\d+)", line)
                if m:
                    mul_active[int(m.group(1))] = int(m.group(2))

                m = re.search(r"mul_units(\d+)\.idleCycles\s+(\d+)", line)
                if m:
                    mul_idle[int(m.group(1))] = int(m.group(2))

                m = re.search(r"acc_units(\d+)\.activeCycles\s+(\d+)", line)
                if m:
                    acc_active[int(m.group(1))] = int(m.group(2))

                m = re.search(r"acc_units(\d+)\.idleCycles\s+(\d+)", line)
                if m:
                    acc_idle[int(m.group(1))] = int(m.group(2))

                m = re.search(r"\bmul\.activeCycles\s+(\d+)", line)
                if m:
                    mul_active[0] = int(m.group(1))

                m = re.search(r"\bmul\.idleCycles\s+(\d+)", line)
                if m:
                    mul_idle[0] = int(m.group(1))

                m = re.search(r"\bacc\.activeCycles\s+(\d+)", line)
                if m:
                    acc_active[0] = int(m.group(1))

                m = re.search(r"\bacc\.idleCycles\s+(\d+)", line)
                if m:
                    acc_idle[0] = int(m.group(1))

        num_pes = len(mul_active)

        mul_utils = []
        acc_utils = []

        for pe in mul_active.keys():

            if pe not in mul_idle:
                continue

            total = mul_active[pe] + mul_idle[pe]

            if total > 0:
                util = mul_active[pe] / total
                mul_utils.append(util)

        for pe in acc_active.keys():

            if pe not in acc_idle:
                continue

            total = acc_active[pe] + acc_idle[pe]

            if total > 0:
                util = acc_active[pe] / total
                acc_utils.append(util)

        if len(mul_utils) == 0 and len(acc_utils) == 0:
            continue

        records.append(
            {
                "model": model,
                "arch": arch.capitalize(),
                "mul_util": np.mean(mul_utils),
                "acc_util": np.mean(acc_utils),
                "mul_std": np.std(mul_utils),
                "acc_std": np.std(acc_utils),
            }
        )

data = pd.DataFrame(records)

if data.empty:
    raise RuntimeError("No stats found")

ARCH_ORDER = ["Sequential", "Parallel", "Gamma", "Sparta"]


avg = data.groupby("arch")[["mul_util", "acc_util"]].mean()
avg = avg.reindex(ARCH_ORDER)

avg.plot(kind="bar", figsize=(8, 6))

plt.ylabel("Average Utilization")
plt.title("Average PE Utilization Across Architectures")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/avg_pe_utilization.png")
plt.close()

print("Saved avg_pe_utilization.png")


pivot_mul = data.pivot(index="model", columns="arch", values="mul_util")
pivot_acc = data.pivot(index="model", columns="arch", values="acc_util")

pivot_mul = pivot_mul.reindex(columns=ARCH_ORDER)
pivot_acc = pivot_acc.reindex(columns=ARCH_ORDER)

pivot_mul.plot(kind="bar", figsize=(12, 6))

plt.ylabel("Mul Utilization")
plt.title("Multiplier Utilization")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/mul_utilization.png")
plt.close()

print("Saved mul_utilization.png")

pivot_acc.plot(kind="bar", figsize=(12, 6))

plt.ylabel("Accumulator Utilization")
plt.title("Accumulator Utilization")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/acc_utilization.png")
plt.close()

print("Saved acc_utilization.png")
print("\nAll PE utilization plots saved in:", OUT_DIR)
