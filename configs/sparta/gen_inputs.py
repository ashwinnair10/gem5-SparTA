import argparse
import os

import numpy as np

# ----------------------------
# Sparsity helpers
# ----------------------------


def random_sparsity(mat, sparsity):
    if sparsity <= 0.0:
        return mat
    mask = np.random.rand(*mat.shape) >= sparsity
    return mat * mask


def block_sparsity(mat, block_size, sparsity):
    if sparsity <= 0.0:
        return mat

    mat = mat.copy()
    rows, cols = mat.shape

    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            if np.random.rand() < sparsity:
                mat[i : i + block_size, j : j + block_size] = 0.0
    return mat


def local_attention_sparsity(X, window):
    seqlen, dmodel = X.shape
    X = X.copy()

    for i in range(seqlen):
        left = max(0, i - window)
        right = i + 1
        X[i, :left] = 0.0
        X[i, right:] = 0.0

    return X


# ----------------------------
# Argument parsing
# ----------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--dmodel", type=int, required=True)
parser.add_argument("--seqlen", type=int, required=True)

parser.add_argument(
    "--sparsity-mode",
    choices=["none", "random", "block", "local"],
    default="none",
)

parser.add_argument(
    "--sparsity",
    type=float,
    default=0.0,
    help="fraction of zeros (used in random/block)",
)

parser.add_argument(
    "--block-size", type=int, default=16, help="block size for block sparsity"
)

parser.add_argument(
    "--window",
    type=int,
    default=32,
    help="attention window for local sparsity",
)

args = parser.parse_args()

# ----------------------------
# Output directory
# ----------------------------

os.makedirs("configs/sparta/inputs", exist_ok=True)

np.random.seed(42)

# ----------------------------
# Generate dense matrices
# ----------------------------

X = np.random.uniform(-1, 1, (args.seqlen, args.dmodel)).astype(np.float32)
W = np.random.uniform(-1, 1, (3 * args.dmodel, args.dmodel)).astype(np.float32)

# ----------------------------
# Apply sparsity
# ----------------------------

if args.sparsity_mode == "random":
    X = random_sparsity(X, args.sparsity)
    W = random_sparsity(W, args.sparsity)

elif args.sparsity_mode == "block":
    X = block_sparsity(X, args.block_size, args.sparsity)
    W = block_sparsity(W, args.block_size, args.sparsity)

elif args.sparsity_mode == "local":
    X = local_attention_sparsity(X, args.window)
    # W usually stays dense for attention workloads

# ----------------------------
# Save
# ----------------------------

np.save("configs/sparta/inputs/X.npy", X)
np.save("configs/sparta/inputs/W.npy", W)

print("Saved matrices to configs/sparta/inputs/")
print(f"Sparsity mode   : {args.sparsity_mode}")
print(f"X sparsity      : {np.mean(X == 0):.3f}")
print(f"W sparsity      : {np.mean(W == 0):.3f}")
