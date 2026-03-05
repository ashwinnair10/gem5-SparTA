import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..")
)
sys.path.insert(0, PROJECT_ROOT)

from configs.sparta.common.attention_math import (
    matmul,
    softmax,
)


def compute_reference(X, W, dmodel):

    QKV = matmul(X, W)

    Q = [r[0 * dmodel : 1 * dmodel] for r in QKV]
    K = [r[1 * dmodel : 2 * dmodel] for r in QKV]
    V = [r[2 * dmodel : 3 * dmodel] for r in QKV]

    K_T = list(zip(*K))

    scores = matmul(Q, K_T)
    probs = softmax(scores)
    out = matmul(probs, V)

    return np.array(out)


def verify(gem5_output_path, X_np, W_np, dmodel, tol=1e-4):

    if not os.path.exists(gem5_output_path):
        raise RuntimeError(f"GEM5 output missing: {gem5_output_path}")

    gem5_out = np.load(gem5_output_path)

    # convert to list only if required by matmul implementation
    X = X_np.tolist()
    W = W_np.tolist()

    ref = compute_reference(X, W, dmodel)

    if gem5_out.shape != ref.shape:
        raise RuntimeError(
            f"Output shape mismatch: gem5 {gem5_out.shape} vs ref {ref.shape}"
        )

    diff = np.abs(gem5_out - ref)

    max_err = diff.max()
    mean_err = diff.mean()

    print("Verification Results:")
    print("  Max error :", max_err)
    print("  Mean error:", mean_err)

    if max_err < tol:
        print("  ✔ PASS")
    else:
        print("  ✘ FAIL")

    return max_err, mean_err
