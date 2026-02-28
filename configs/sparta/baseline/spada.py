# configs/sparta/baseline/spada.py
import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..")
)
sys.path.insert(0, PROJECT_ROOT)

from configs.sparta.common.attention_setup import setup_attention
from configs.sparta.common.input_loader import load_X_W
from configs.sparta.common.pe_factory import make_pes
from configs.sparta.drivers.spada import attach_driver

import m5

X, W, seqlen, dmodel = load_X_W(
    "configs/sparta/inputs/X.npy", "configs/sparta/inputs/W.npy"
)

root, *mats = setup_attention(X, W, seqlen, dmodel)

parser = argparse.ArgumentParser()

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
pes = make_pes(
    args.numPEs,
    args.mulLatency,
    args.accLatency,
    args.mulQueueSize,
    args.accQueueSize,
)

attach_driver(root, pes, args, mats, (seqlen, dmodel))

m5.instantiate()
m5.simulate()

# ---------------- SAVE GEM5 OUTPUT ----------------
import numpy as np

# mats returned from setup_attention:
# mats = (Q_m, K_m, V_m, Scores_m, Prob_m, Output_m)

Output_m = mats[-1]  # last matrix is Output
seq_len, d_model = seqlen, dmodel

from configs.sparta.common.shared_mem import shared_to_list

Output = shared_to_list(Output_m, seq_len, d_model)

outdir = m5.options.outdir
np.save(os.path.join(outdir, "gem5_output.npy"), np.array(Output))

print("[INFO] Saved gem5_output.npy")
