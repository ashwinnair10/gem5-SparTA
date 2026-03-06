import argparse
import os
import shutil

import numpy as np


def prune_matrix(W, sparsity):
    """
    Magnitude pruning
    sparsity = fraction of weights to remove
    """
    W_pruned = W.copy()

    threshold = np.percentile(np.abs(W_pruned), sparsity * 100)
    W_pruned[np.abs(W_pruned) < threshold] = 0

    return W_pruned


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="input model folder inside configs/sparta/inputs",
    )
    parser.add_argument(
        "--sparsities",
        nargs="+",
        type=float,
        default=[0.5, 0.8, 0.9],
        help="sparsity levels",
    )

    args = parser.parse_args()

    base_dir = f"configs/sparta/inputs/{args.model}"

    X_path = os.path.join(base_dir, "X.npy")
    W_path = os.path.join(base_dir, "W.npy")

    if not os.path.exists(W_path):
        raise RuntimeError(f"W.npy not found in {base_dir}")

    print("Loading weights...")
    W = np.load(W_path)
    X = np.load(X_path)

    for s in args.sparsities:

        print(f"Pruning to {int(s*100)}% sparsity")

        W_pruned = prune_matrix(W, s)

        out_dir = f"configs/sparta/inputs/{args.model}-{int(s*100)}-sparse"

        os.makedirs(out_dir, exist_ok=True)

        np.save(os.path.join(out_dir, "W.npy"), W_pruned)
        np.save(os.path.join(out_dir, "X.npy"), X)

        density = np.count_nonzero(W_pruned) / W_pruned.size

        print(f"Saved {out_dir}")
        print(f"Density: {density:.4f}")


if __name__ == "__main__":
    main()
