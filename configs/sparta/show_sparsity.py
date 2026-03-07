import sys

import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) != 2:
    print("Usage:")
    print("  python3 visualize_sparsity_npy.py matrix.npy")
    sys.exit(1)

file = sys.argv[1]
mat = np.load(file)
mask = mat != 0

sparsity = 1.0 - np.count_nonzero(mat) / mat.size

print("Shape     :", mat.shape)
print("Nonzeros  :", np.count_nonzero(mat))
print("Sparsity  :", sparsity)

plt.figure(figsize=(6, 6))
plt.imshow(mask, cmap="gray_r", interpolation="nearest")
plt.title(f"Sparsity pattern ({sparsity:.2%} zeros)")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.tight_layout()
plt.show()
