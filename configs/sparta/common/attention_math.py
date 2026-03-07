import math


def matmul(a, b):
    return [
        [
            sum(a[i][k] * b[k][j] for k in range(len(a[0])))
            for j in range(len(b[0]))
        ]
        for i in range(len(a))
    ]


def softmax(mat):
    out = []
    for row in mat:
        m = max(row)
        exps = [math.exp(x - m) for x in row]
        s = sum(exps)
        out.append([x / s for x in exps])
    return out
