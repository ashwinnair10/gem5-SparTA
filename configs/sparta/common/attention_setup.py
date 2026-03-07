import numpy as np
from numpy import shape
from sympy import root

from m5.objects import Root

from .attention_math import matmul
from .shared_mem import (
    alloc_shared_matrix,
    list_to_shared,
)


def attention(X, W, seqlen, dmodel):

    WQ = [r[0 * dmodel : 1 * dmodel] for r in W]
    WK = [r[1 * dmodel : 2 * dmodel] for r in W]
    WV = [r[2 * dmodel : 3 * dmodel] for r in W]

    X_m, X_base, X_mm = list_to_shared(X)
    WQ_m, WQ_base, WQ_mm = list_to_shared(WQ)
    WK_m, WK_base, WK_mm = list_to_shared(WK)
    WV_m, WV_base, WV_mm = list_to_shared(WV)

    Q_m, Q_base, Q_mm = alloc_shared_matrix(seqlen, dmodel)
    K_m, K_base, K_mm = alloc_shared_matrix(seqlen, dmodel)
    V_m, V_base, V_mm = alloc_shared_matrix(seqlen, dmodel)

    Scores_m, Scores_base, Scores_mm = alloc_shared_matrix(seqlen, seqlen)
    Prob_m, Prob_base, Prob_mm = alloc_shared_matrix(seqlen, seqlen)
    Output_m, Output_base, Output_mm = alloc_shared_matrix(seqlen, dmodel)

    root = Root(full_system=False)

    root._mmaps = [
        X_mm,
        WQ_mm,
        WK_mm,
        WV_mm,
        Q_mm,
        K_mm,
        V_mm,
        Scores_mm,
        Prob_mm,
        Output_mm,
        X_base,
        WQ_base,
        WK_base,
        WV_base,
        Q_base,
        K_base,
        V_base,
        Scores_base,
        Prob_base,
        Output_base,
    ]

    return (
        root,
        X_m,
        WQ_m,
        WK_m,
        WV_m,
        Q_m,
        K_m,
        V_m,
        Scores_m,
        Prob_m,
        Output_m,
    )
