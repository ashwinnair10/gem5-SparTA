# configs/sparta/drivers/spada.py
import ctypes

from m5.objects import AcceleratorDriver


def attach_driver(root, pes, args, mats, dims):
    muls, accs = pes
    X, WQ, WK, WV, Q, K, V, Scores, Prob, Output = mats
    seqlen, dmodel = dims

    root.drv = AcceleratorDriver(
        numPEs=args.numPEs,
        mul_units=muls,
        acc_units=accs,
        mul_queue_depth=args.mulQueueSize,
        acc_queue_depth=args.accQueueSize,
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
