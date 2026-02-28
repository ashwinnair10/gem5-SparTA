import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..")
)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np

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


def verify(gem5_output_path, X_np, W_np, dmodel):
    gem5_out = np.load(gem5_output_path)

    X = X_np.tolist()
    W = W_np.tolist()

    ref = compute_reference(X, W, dmodel)

    diff = np.abs(gem5_out - ref)

    # print("\nGEM5 Output:\n")
    # for row in gem5_out:
    #     print("[{}]".format(" ".join(f"{x:.4f}" for x in row)))
    # print("\nReference Output:\n")
    # for row in ref:
    #     print("[{}]".format(" ".join(f"{x:.4f}" for x in row)))
    # print("\nDifference:\n")
    # for row in diff:
    #     print("[{}]".format(" ".join(f"{x:.4f}" for x in row)))

    max_err = diff.max()
    mean_err = diff.mean()

    print("Verification Results:")
    print("  Max error :", max_err)
    print("  Mean error:", mean_err)

    if max_err < 1e-4:
        print("  ✔ PASS")
    else:
        print("  ✘ FAIL")

    return max_err, mean_err
