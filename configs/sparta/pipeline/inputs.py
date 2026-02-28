# configs/sparta/pipeline/inputs.py
import os
import subprocess

import numpy as np

GEN_INPUTS = "configs/sparta/gen_inputs.py"


def prepare_inputs(args):
    if args.X and args.W:
        return _use_real_inputs(args)
    else:
        return _use_generated_inputs(args)


def _use_real_inputs(args):
    print("[INPUT] Loading real matrices from:" f" {args.X} and {args.W}")
    if not (args.X and args.W):
        raise ValueError("Both --X and --W required")

    X = np.load(args.X)
    seqlen, dmodel = X.shape

    os.makedirs("configs/sparta/inputs", exist_ok=True)
    # subprocess.run(["cp", args.X, "configs/sparta/inputs/X.npy"], check=True)
    # subprocess.run(["cp", args.W, "configs/sparta/inputs/W.npy"], check=True)

    print("[INPUT] Using real matrices")
    print(f"  seqlen={seqlen}, dmodel={dmodel}")

    return seqlen, dmodel


def _use_generated_inputs(args):
    if args.dmodel is None or args.seqlen is None:
        raise ValueError("--dmodel and --seqlen required for synthetic inputs")

    cmd = [
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
        cmd += ["--sparsity", str(args.sparsity)]
    if args.sparsity_mode == "block":
        cmd += ["--block-size", str(args.block_size)]
    if args.sparsity_mode == "local":
        cmd += ["--window", str(args.window)]

    subprocess.run(cmd, check=True)

    print("[INPUT] Using synthetic matrices")

    return args.seqlen, args.dmodel
