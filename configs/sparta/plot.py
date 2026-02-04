import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("dir", help="stats/<dmodel>_<seqlen>_<numpes>_<mode>")
args = parser.parse_args()

mode = args.dir.split("_")[-1]  # random / block / local
xlabel = "Window Size" if mode == "local" else "Sparsity (%)"

records = []

for fname in sorted(
    os.listdir(args.dir), key=lambda x: float(x.replace(".csv", ""))
):
    if not fname.endswith(".csv"):
        continue

    xval = float(fname.replace(".csv", ""))
    xplot = xval if mode == "local" else xval * 100

    df = pd.read_csv(os.path.join(args.dir, fname), index_col=0)

    for design in ["Sequential", "Parallel", "SparTA"]:
        records.append(
            {
                "X": xplot,
                "Design": design,
                "simTicks": df.loc[design, "simTicks"],
                "Speedup": df.loc[design, "Speedup"],
                "Energy": df.loc[design, "runtime_dynamic"],
                "EDP": df.loc[design, "EDP"],
                "EDPNorm": df.loc[design, "EDPNorm_vs_SparTA"],
            }
        )

data = pd.DataFrame(records)


# -------- helper --------
def plot(metric, ylabel, fname):
    plt.figure()
    for d in ["Sequential", "Parallel", "SparTA"]:
        sub = data[data.Design == d]
        plt.plot(sub.X, sub[metric], marker="o", label=d)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.savefig(fname, dpi=300)
    plt.close()


# -------- plots --------
plot("Speedup", "Speedup (vs Sequential)", "speedup.png")
plot("simTicks", "Simulation Ticks", "time.png")
plot("Energy", "Runtime Dynamic Energy (McPAT)", "energy.png")
plot("EDPNorm", "EDP (Normalized to SparTA)", "edp.png")

print("Generated:")
print("  speedup.png")
print("  time.png")
print("  energy.png")
print("  edp.png")
