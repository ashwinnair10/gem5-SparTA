# configs/sparta/run_all.py
import argparse
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..")
)
sys.path.insert(0, PROJECT_ROOT)
import argparse
import os

from dotenv import load_dotenv

from configs.sparta.pipeline.gem5_runner import run_test
from configs.sparta.pipeline.inputs import prepare_inputs
from configs.sparta.pipeline.mcpat import (
    generate_mcpat_xml,
    parse_mcpat,
    run_mcpat,
)
from configs.sparta.pipeline.report import (
    build_report,
    save_csv,
)
from configs.sparta.pipeline.stats import extract_simticks
from configs.sparta.pipeline.verification import verify

load_dotenv()

TESTS = [
    ("Sequential", "configs/sparta/baseline/sequential_test.py"),
    # ("Parallel",   "configs/sparta/baseline/parallel_test.py"),
    ("SparTA", "configs/sparta/accelerator/accelerator_test.py"),
    # ("SPADA",      "configs/sparta/baseline/spada.py"),
    # ("Block-SPADA","configs/sparta/baseline/blockspada.py"),
    ("Gamma", "configs/sparta/baseline/gamma.py"),
]

parser = argparse.ArgumentParser()

# model / execution params
parser.add_argument("-d", "--dmodel", type=int)
parser.add_argument("-s", "--seqlen", type=int)
parser.add_argument("-n", "--numPEs", type=int, required=True)

parser.add_argument("--mulQueueSize", type=int, required=True)
parser.add_argument("--accQueueSize", type=int, required=True)
parser.add_argument("--mulLatency", type=int, default=4)
parser.add_argument("--accLatency", type=int, default=2)

# input selection
parser.add_argument("--X", type=str, help="Path to X.npy")
parser.add_argument("--W", type=str, help="Path to W.npy")
parser.add_argument("--use-gen-inputs", action="store_true")

# sparsity (only for gen_inputs)
parser.add_argument(
    "--sparsity-mode",
    choices=["none", "random", "block", "local"],
    default="none",
)
parser.add_argument("--sparsity", type=float, default=0.0)
parser.add_argument("--block-size", type=int, default=16)
parser.add_argument("--window", type=int, default=32)

args = parser.parse_args()


seqlen, dmodel = prepare_inputs(args)

results = {}

for name, script in TESTS:
    stats_path = run_test(name, script, f"stats/{name.lower()}", args)
    ticks = extract_simticks(stats_path)

    X_np = np.load(args.X if args.X else "configs/sparta/inputs/X.npy")
    W_np = np.load(args.W if args.W else "configs/sparta/inputs/W.npy")

    verify(
        os.path.join(f"stats/{name.lower()}", "gem5_output.npy"),
        X_np,
        W_np,
        dmodel,
    )

    mcxml = f"stats/{name.lower()}/mcpat.xml"
    mcout = f"stats/{name.lower()}/mcpat.out"
    generate_mcpat_xml(stats_path, os.environ["MCPAT_BASE_XML"], mcxml)
    run_mcpat(mcxml, mcout)
    energy = parse_mcpat(mcout)

    results[name] = {"simTicks": ticks, **energy}

df = build_report(results)
print(df)

save_csv(df, f"stats/{dmodel}_{seqlen}_{args.numPEs}", "results.csv")
