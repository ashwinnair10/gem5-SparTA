import argparse
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dmodel", type=int, required=True)
parser.add_argument("--seqlen", type=int, required=True)
args = parser.parse_args()

os.makedirs("configs/sparta/inputs", exist_ok=True)

np.random.seed(42)

X = np.random.uniform(-1, 1, (args.seqlen, args.dmodel)).astype(np.float32)
W = np.random.uniform(-1, 1, (3 * args.dmodel, args.dmodel)).astype(np.float32)

np.save("configs/sparta/inputs/X.npy", X)
np.save("configs/sparta/inputs/W.npy", W)

print("Saved matrices to configs/sparta/inputs/")
