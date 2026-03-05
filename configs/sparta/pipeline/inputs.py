# configs/sparta/pipeline/inputs.py

import os
import subprocess
import sys

import numpy as np

GEN_INPUTS = "configs/sparta/gen_inputs.py"


def prepare_inputs(args):

    if args.use_gen_inputs:
        return _use_generated_inputs(args)

    if args.X and args.W:
        return _use_real_inputs(args)

    raise ValueError(
        "Provide (--X and --W) or use --use-gen-inputs with --dmodel and --seqlen"
    )


def _use_real_inputs(args):

    print(f"[INPUT] Loading matrices:\n  X={args.X}\n  W={args.W}")

    X = np.load(args.X)
    W = np.load(args.W)

    seqlen, dmodel = X.shape

    if W.shape[0] != dmodel:
        raise ValueError(
            f"W dimension mismatch: expected first dim {dmodel}, got {W.shape[0]}"
        )

    print("[INPUT] Using real matrices")
    print(f"  seqlen={seqlen}, dmodel={dmodel}")

    return seqlen, dmodel


def _use_generated_inputs(args):

    if args.dmodel is None or args.seqlen is None:
        raise ValueError("--dmodel and --seqlen required for synthetic inputs")

    cmd = [
        sys.executable,
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

    print("[INPUT] Generating synthetic matrices")
    print(">>>", " ".join(cmd))

    subprocess.run(cmd, check=True)

    print("[INPUT] Using synthetic matrices")
    print(f"  seqlen={args.seqlen}, dmodel={args.dmodel}")

    return args.seqlen, args.dmodel
