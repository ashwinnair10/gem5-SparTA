import numpy as np


def load_X_W(X_path, W_path):
    X = np.load(X_path)
    W = np.load(W_path)

    assert X.ndim == 2
    assert W.ndim == 2

    seqlen, dmodel = X.shape
    assert W.shape == (dmodel, 3 * dmodel)

    print("[INFO] Auto-detected:")
    print(f"  seqlen = {seqlen}")
    print(f"  dmodel = {dmodel}")

    return X.tolist(), W.tolist(), seqlen, dmodel
