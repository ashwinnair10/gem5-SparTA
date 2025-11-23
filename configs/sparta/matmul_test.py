import m5
from m5.objects import *

root = Root(full_system=False)

import ctypes

M = 2
N = 2
K = 2

A = (ctypes.POINTER(ctypes.c_float) * M)()
B = (ctypes.POINTER(ctypes.c_float) * K)()
C = (ctypes.POINTER(ctypes.c_float) * M)()

for i in range(M):
    A[i] = (ctypes.c_float * K)()
for i in range(K):
    B[i] = (ctypes.c_float * N)()
for i in range(M):
    C[i] = (ctypes.c_float * N)()

A[0][0] = 1
A[0][1] = 2
A[1][0] = 3
A[1][1] = 4

B[0][0] = 5
B[0][1] = 6
B[1][0] = 7
B[1][1] = 8

root.mul = PE_Mul(latency=10)
root.acc = PE_Acc(latency=5)

root.mm = MatMul(
    mul=root.mul,
    acc=root.acc,
    A=0,
    B=0,
    C=0,
    M=0,
    N=0,
    K=0,
)

m5.instantiate()

print("Starting simulation…")
root.mm.startMatMul(
    A=ctypes.cast(A, ctypes.c_void_p).value,
    B=ctypes.cast(B, ctypes.c_void_p).value,
    C=ctypes.cast(C, ctypes.c_void_p).value,
    M=M,
    N=N,
    K=K,
)
exit_event = m5.simulate()
print("Exit at tick:", m5.curTick(), exit_event.getCause())

print("\nRESULT MATRIX C:")
for i in range(M):
    print([C[i][j] for j in range(N)])
