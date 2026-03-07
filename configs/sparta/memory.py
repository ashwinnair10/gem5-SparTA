import argparse
import math


def tag_memory_comparison(M, N, K, numPEs, mul_q, acc_q):

    mac_ops = M * N * K

    full_tags = M * N
    full_tag_bits = math.ceil(math.log2(full_tags))
    full_total_bits = mac_ops * full_tag_bits

    max_live_ops = numPEs * (mul_q + acc_q + 1) * 4
    sid_bits = math.ceil(math.log2(max_live_ops))
    sid_total_bits = max_live_ops * sid_bits

    return {
        "mac_ops": mac_ops,
        "full": {
            "unique_tags": full_tags,
            "bits_per_tag": full_tag_bits,
            "total_MB": full_total_bits / 8 / (1024),
        },
        "sid": {
            "max_live_ops": max_live_ops,
            "bits_per_tag": sid_bits,
            "total_MB": sid_total_bits / 8 / (1024),
        },
        "reduction": {
            "memory_saved_MB": (full_total_bits - sid_total_bits) / 8 / (1024),
            "reduction_factor": full_total_bits / sid_total_bits,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Tag memory cost using d_model and seq_len"
    )

    parser.add_argument(
        "--dmodel", type=int, required=True, help="Model dimension (d_model)"
    )
    parser.add_argument(
        "--seqlen", type=int, required=True, help="Sequence length"
    )

    parser.add_argument("--pes", type=int, required=True, help="Number of PEs")
    parser.add_argument(
        "--mulq", type=int, required=True, help="Mul queue depth"
    )
    parser.add_argument(
        "--accq", type=int, required=True, help="Acc queue depth"
    )

    args = parser.parse_args()

    M = args.seqlen
    N = args.seqlen
    K = args.dmodel

    stats = tag_memory_comparison(M, N, K, args.pes, args.mulq, args.accq)

    print(f"d_model  = {args.dmodel}")
    print(f"seq_len  = {args.seqlen}")

    print("\nDerived GEMM:")
    print(f"M = seq_len = {M}")
    print(f"N = seq_len = {N}")
    print(f"K = d_model = {K}")

    print(f"\nTotal MAC operations : {stats['mac_ops']:,}")

    print("\n Full (i·N + j) tagging ")
    print(f"Unique tags          : {stats['full']['unique_tags']:,}")
    print(f"Bits per tag         : {stats['full']['bits_per_tag']}")
    print(f"Total tag memory     : {stats['full']['total_MB']} KB")

    print("\n SID tagging ")
    print(f"Max live SIDs        : {stats['sid']['max_live_ops']}")
    print(f"Bits per SID         : {stats['sid']['bits_per_tag']}")
    print(f"Total tag memory     : {stats['sid']['total_MB']} KB")

    print("\n Savings ")
    print(f"Memory saved         : {stats['reduction']['memory_saved_MB']} KB")
    print(f"Reduction factor     : {stats['reduction']['reduction_factor']}×")


if __name__ == "__main__":
    main()
