from m5.objects import (
    PE_Acc,
    PE_Mul,
)


def make_pes(num, mul_lat, acc_lat, mul_q, acc_q):
    muls = [
        PE_Mul(latency=mul_lat, island=i, queue_size=mul_q) for i in range(num)
    ]
    accs = [
        PE_Acc(latency=acc_lat, island=i, queue_size=acc_q) for i in range(num)
    ]
    return muls, accs
