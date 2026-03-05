# configs/sparta/run_all.py

import argparse
import os
import sys

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..")
)
sys.path.insert(0, PROJECT_ROOT)

from configs.sparta.pipeline.gem5_runner import run_test
from configs.sparta.pipeline.inputs import prepare_inputs
from configs.sparta.pipeline.report import (
    build_report,
    save_csv,
)
from configs.sparta.pipeline.stats import extract_simticks
from configs.sparta.pipeline.verification import verify

load_dotenv()


TESTS = [
    ("Sequential", "configs/sparta/baseline/sequential_test.py"),
    ("Parallel", "configs/sparta/baseline/parallel_test.py"),
    ("SparTA", "configs/sparta/accelerator/accelerator_test.py"),
    ("Gamma", "configs/sparta/baseline/gamma.py"),
]


parser = argparse.ArgumentParser()

# model parameters
parser.add_argument("-d", "--dmodel", type=int)
parser.add_argument("-s", "--seqlen", type=int)
parser.add_argument("-n", "--numPEs", type=int, required=True)

# accelerator parameters
parser.add_argument("--mulQueueSize", type=int, required=True)
parser.add_argument("--accQueueSize", type=int, required=True)
parser.add_argument("--mulLatency", type=int, default=4)
parser.add_argument("--accLatency", type=int, default=2)

# workload identification (important for batch)
parser.add_argument("--model", type=str, default="default")

# input selection
parser.add_argument("--X", type=str)
parser.add_argument("--W", type=str)
parser.add_argument("--use-gen-inputs", action="store_true")

# sparsity (gen_inputs)
parser.add_argument(
    "--sparsity-mode",
    choices=["none", "random", "block", "local"],
    default="none",
)
parser.add_argument("--sparsity", type=float, default=0.0)
parser.add_argument("--block-size", type=int, default=16)
parser.add_argument("--window", type=int, default=32)

args = parser.parse_args()


# ------------------------------------------------
# Prepare inputs
# ------------------------------------------------

seqlen, dmodel = prepare_inputs(args)

X_np = np.load(args.X if args.X else "configs/sparta/inputs/X.npy")
W_np = np.load(args.W if args.W else "configs/sparta/inputs/W.npy")

results = {}

# ------------------------------------------------
# Run architectures
# ------------------------------------------------

for name, script in TESTS:

    stats_dir = f"stats/{args.model}/{name.lower()}"

    stats_path = run_test(name, script, stats_dir, args)

    ticks = extract_simticks(os.path.join(stats_path, "stats.txt"))

    verify(
        os.path.join(stats_path, "gem5_output.npy"),
        X_np,
        W_np,
        dmodel,
    )

    results[name] = {"simTicks": ticks}


# ------------------------------------------------
# Build report
# ------------------------------------------------

df = build_report(results)

print(df)

save_csv(
    df,
    f"stats/{args.model}",
    "results.csv",
)
