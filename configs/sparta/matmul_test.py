from m5.objects import *

root = Root(full_system=False)
root.system = System()

system = root.system
system.clk_domain = SrcClockDomain(
    clock="1GHz", voltage_domain=VoltageDomain()
)
system.mem_mode = "timing"
system.mem_ranges = [AddrRange("1MB")]

import ctypes

M = 2
N = 2
K = 2

A = (ctypes.POINTER(ctypes.c_int) * M)()
B = (ctypes.POINTER(ctypes.c_int) * K)()
C = (ctypes.POINTER(ctypes.c_int) * M)()

for i in range(M):
    A[i] = (ctypes.c_int * K)()
for i in range(K):
    B[i] = (ctypes.c_int * N)()
for i in range(M):
    C[i] = (ctypes.c_int * N)()

A[0][0] = 1
A[0][1] = 2
A[1][0] = 3
A[1][1] = 4

B[0][0] = 5
B[0][1] = 6
B[1][0] = 7
B[1][1] = 8

system.mul = PE_Mul(latency=10)
system.acc = PE_Acc(latency=5)

system.mm = MatMul(
    mul=system.mul,
    acc=system.acc,
    A=ctypes.cast(A, ctypes.c_void_p).value,
    B=ctypes.cast(B, ctypes.c_void_p).value,
    C=ctypes.cast(C, ctypes.c_void_p).value,
    M=M,
    N=N,
    K=K,
)

m5.instantiate()

print("Starting simulation…")
exit_event = m5.simulate()
print("Exit at tick:", m5.curTick(), exit_event.getCause())

print("\nRESULT MATRIX C:")
for i in range(M):
    print([C[i][j] for j in range(N)])
