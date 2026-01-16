import argparse
import os
import subprocess

import pandas as pd

GEM5 = "build/ARM/gem5.opt"
STATS_ROOT = "stats"
GEN_INPUTS = "configs/sparta/gen_inputs.py"

TESTS = [
    ("Sequential", "configs/sparta/baseline/sequential_test.py"),
    ("Parallel", "configs/sparta/baseline/parallel_test.py"),
    ("SparTA", "configs/sparta/accelerator/accelerator_test.py"),
]


def run_cmd(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_test(name, script, outdir, args):
    os.makedirs(outdir, exist_ok=True)

    cmd = [
        GEM5,
        f"--outdir={outdir}",
        script,
        "--dmodel",
        str(args.dmodel),
        "--seqlen",
        str(args.seqlen),
        "--queueSize",
        str(args.queueSize),
    ]

    if name != "Sequential":
        cmd += ["--numPEs", str(args.numPEs)]

    run_cmd(cmd)


def extract_simticks(stats_path):
    with open(stats_path) as f:
        for line in f:
            if line.startswith("simTicks"):
                return int(line.split()[1])
    raise RuntimeError(f"simTicks not found in {stats_path}")


# ---------------- args ----------------
parser = argparse.ArgumentParser()

# model params
parser.add_argument("-d", "--dmodel", type=int, required=True)
parser.add_argument("-s", "--seqlen", type=int, required=True)
parser.add_argument("-n", "--numPEs", type=int, required=True)
parser.add_argument("-q", "--queueSize", type=int, required=True)

# sparsity params (forwarded to gen_inputs.py)
parser.add_argument(
    "--sparsity-mode",
    choices=["none", "random", "block", "local"],
    default="none",
)

parser.add_argument("--sparsity", type=float, default=0.0)
parser.add_argument("--block-size", type=int, default=16)
parser.add_argument("--window", type=int, default=32)

args = parser.parse_args()

# ---------------- prepare stats dir ----------------
os.makedirs(STATS_ROOT, exist_ok=True)

# ---------------- generate shared inputs ----------------
print("\n===== Generating shared input matrices =====")

gen_cmd = [
    "python3",
    GEN_INPUTS,
    "--dmodel",
    str(args.dmodel),
    "--seqlen",
    str(args.seqlen),
    "--sparsity-mode",
    args.sparsity_mode,
]

if args.sparsity_mode in ("random", "block"):
    gen_cmd += ["--sparsity", str(args.sparsity)]

if args.sparsity_mode == "block":
    gen_cmd += ["--block-size", str(args.block_size)]

if args.sparsity_mode == "local":
    gen_cmd += ["--window", str(args.window)]

run_cmd(gen_cmd)

# ---------------- run all tests ----------------
results = {}

for name, script in TESTS:
    outdir = os.path.join(STATS_ROOT, name.lower())
    print(f"\n===== Running {name} =====")
    run_test(name, script, outdir, args)

    stats_file = os.path.join(outdir, "stats.txt")
    ticks = extract_simticks(stats_file)
    results[name] = ticks

# ---------------- report ----------------
df = pd.DataFrame.from_dict(results, orient="index", columns=["simTicks"])
df["SparTA Speedup"] = df["simTicks"] / df.loc["SparTA", "simTicks"]

print("\n=== PERFORMANCE COMPARISON ===")
print(df)

csv_path = os.path.join(STATS_ROOT, "sparta_comparison.csv")
df.to_csv(csv_path)
print(f"\nSaved {csv_path}")
