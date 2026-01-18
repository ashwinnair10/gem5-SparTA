import argparse
import ctypes
import math
import mmap
import random

import numpy as np

import m5
from m5.objects import *


def rand_mat(rows, cols):
    return [[random.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]


def zeros(rows, cols):
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def matmul(a, b):
    m = len(a)
    k = len(a[0])
    n = len(b[0])
    out = zeros(m, n)
    for i in range(m):
        for j in range(n):
            s = 0
            for t in range(k):
                s += a[i][t] * b[t][j]
            out[i][j] = s
    return out


def softmax_row(v):
    mx = max(v)
    exps = [math.exp(x - mx) for x in v]
    s = sum(exps)
    return [x / s for x in exps]


def softmax(mat):
    return [softmax_row(row) for row in mat]


def alloc_shared_matrix(rows, cols):
    total = rows * cols * ctypes.sizeof(ctypes.c_float)
    mm = mmap.mmap(-1, total)
    base = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(mm)))

    RowArray = ctypes.POINTER(ctypes.c_float) * rows
    matrix = RowArray()

    for i in range(rows):
        row_addr = base.value + i * cols * ctypes.sizeof(ctypes.c_float)
        matrix[i] = ctypes.cast(row_addr, ctypes.POINTER(ctypes.c_float))

    return matrix, base, mm


def list_to_shared(mat):
    rows = len(mat)
    cols = len(mat[0])
    m, base, mm = alloc_shared_matrix(rows, cols)
    for i in range(rows):
        for j in range(cols):
            m[i][j] = float(mat[i][j])
    return m, base, mm


def shared_to_list(m, rows, cols):
    out = zeros(rows, cols)
    for i in range(rows):
        for j in range(cols):
            out[i][j] = float(m[i][j])
    return out


parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--dmodel", type=int, required=True, help="model dimension"
)
parser.add_argument(
    "-s", "--seqlen", type=int, required=True, help="sequence length"
)
parser.add_argument(
    "-n", "--numPEs", type=int, required=True, help="number of PES"
)
parser.add_argument(
    "--mulQueueSize", type=int, required=True, help="queue size for Mul PEs"
)
parser.add_argument(
    "--accQueueSize", type=int, required=True, help="queue size for Acc PEs"
)
args = parser.parse_args()

d_model = args.dmodel
seq_len = args.seqlen
num = args.numPEs
mul_queue_size = args.mulQueueSize
acc_queue_size = args.accQueueSize

X = np.load("configs/sparta/inputs/X.npy").tolist()
W = np.load("configs/sparta/inputs/W.npy").tolist()
b = [0.0] * (3 * d_model)

QKV = matmul(X, [list(col) for col in zip(*W)])
Q = [row[0 * d_model : 1 * d_model] for row in QKV]
K = [row[1 * d_model : 2 * d_model] for row in QKV]
V = [row[2 * d_model : 3 * d_model] for row in QKV]
K_T = [list(col) for col in zip(*K)]

Q_m, Q_base, Q_mm = list_to_shared(Q)
K_m, K_base, K_mm = list_to_shared(K_T)
V_m, V_base, V_mm = list_to_shared(V)

Scores_m, Scores_base, Scores_mm = alloc_shared_matrix(seq_len, seq_len)
Prob_m, Prob_base, Prob_mm = alloc_shared_matrix(seq_len, seq_len)
Output_m, Output_base, Output_mm = alloc_shared_matrix(seq_len, d_model)

root = Root(full_system=False)

mul_list = [
    PE_Mul(latency=5, island=i, queue_size=mul_queue_size) for i in range(num)
]
acc_list = [
    PE_Acc(latency=3, island=i, queue_size=acc_queue_size) for i in range(num)
]

root.drv = BaselineDriverParallel(
    numPEs=num,
    mul_units=mul_list,
    acc_units=acc_list,
    Q=ctypes.cast(Q_m, ctypes.c_void_p).value,
    K=ctypes.cast(K_m, ctypes.c_void_p).value,
    V=ctypes.cast(V_m, ctypes.c_void_p).value,
    Scores=ctypes.cast(Scores_m, ctypes.c_void_p).value,
    Prob=ctypes.cast(Prob_m, ctypes.c_void_p).value,
    Output=ctypes.cast(Output_m, ctypes.c_void_p).value,
    M=seq_len,
    N=seq_len,
    Kdim=d_model,
)

m5.instantiate()

print("Running baseline attention inside gem5...")
m5.simulate()

Output = shared_to_list(Output_m, seq_len, d_model)

Ref1 = matmul(Q, [list(col) for col in zip(*K)])
RefSoft = softmax(Ref1)
RefOut = matmul(RefSoft, V)

print("\nGEM5 Output:")
for r in Output:
    print([round(i, 4) for i in r])

print("\nReference Output:")
for r in RefOut:
    print([round(i, 4) for i in r])

print("\nDifference:")
for i in range(seq_len):
    print([round(Output[i][j] - RefOut[i][j], 4) for j in range(d_model)])
