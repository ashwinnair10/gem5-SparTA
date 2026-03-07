import ctypes

from m5.objects import (
    BaselineDriverSequential,
    MatMul,
    PE_Acc,
    PE_Mul,
)


def attach_driver(root, args, mats, dims):
    X, WQ, WK, WV, Q, K, V, Scores, Prob, Output = mats
    seqlen, dmodel = dims

    root.mul = PE_Mul(latency=args.mulLatency, queue_size=args.mulQueueSize)
    root.acc = PE_Acc(latency=args.accLatency, queue_size=args.accQueueSize)
    root.mm = MatMul(mul=root.mul, acc=root.acc)

    root.drv = BaselineDriverSequential(
        mm=root.mm,
        X=ctypes.cast(X, ctypes.c_void_p).value,
        WQ=ctypes.cast(WQ, ctypes.c_void_p).value,
        WK=ctypes.cast(WK, ctypes.c_void_p).value,
        WV=ctypes.cast(WV, ctypes.c_void_p).value,
        Q=ctypes.cast(Q, ctypes.c_void_p).value,
        K=ctypes.cast(K, ctypes.c_void_p).value,
        V=ctypes.cast(V, ctypes.c_void_p).value,
        Scores=ctypes.cast(Scores, ctypes.c_void_p).value,
        Prob=ctypes.cast(Prob, ctypes.c_void_p).value,
        Output=ctypes.cast(Output, ctypes.c_void_p).value,
        M=seqlen,
        N=seqlen,
        Kdim=dmodel,
    )
