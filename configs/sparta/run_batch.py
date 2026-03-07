import argparse
import os
import subprocess
import sys

RUN_SCRIPT = "configs/sparta/run_all.py"
INPUT_ROOT = "configs/sparta/inputs/test"


parser = argparse.ArgumentParser()

parser.add_argument("-n", "--numPEs", type=int, required=True)

parser.add_argument("--mulQueueSize", type=int, required=True)
parser.add_argument("--accQueueSize", type=int, required=True)

parser.add_argument("--mulLatency", type=int, default=4)
parser.add_argument("--accLatency", type=int, default=2)

parser.add_argument("-d", "--dmodel", type=int)
parser.add_argument("-s", "--seqlen", type=int)

parser.add_argument("--use-gen-inputs", action="store_true")

parser.add_argument(
    "--sparsity-mode",
    choices=["none", "random", "block", "local"],
    default="none",
)

parser.add_argument("--sparsity", type=float, default=0.0)
parser.add_argument("--block-size", type=int, default=16)
parser.add_argument("--window", type=int, default=32)

args = parser.parse_args()


def build_base_cmd():

    cmd = [
        sys.executable,
        RUN_SCRIPT,
        "-n",
        str(args.numPEs),
        "--mulQueueSize",
        str(args.mulQueueSize),
        "--accQueueSize",
        str(args.accQueueSize),
        "--mulLatency",
        str(args.mulLatency),
        "--accLatency",
        str(args.accLatency),
        "--sparsity-mode",
        args.sparsity_mode,
        "--sparsity",
        str(args.sparsity),
        "--block-size",
        str(args.block_size),
        "--window",
        str(args.window),
    ]

    if args.use_gen_inputs:
        cmd.append("--use-gen-inputs")

    if args.dmodel:
        cmd += ["-d", str(args.dmodel)]

    if args.seqlen:
        cmd += ["-s", str(args.seqlen)]

    return cmd


def run_model(model_dir):

    x_path = os.path.join(model_dir, "X.npy")
    w_path = os.path.join(model_dir, "W.npy")

    if not (os.path.exists(x_path) and os.path.exists(w_path)):
        print(f"Skipping {model_dir} (missing X/W)")
        return

    model_name = os.path.basename(model_dir)

    cmd = build_base_cmd()

    cmd += ["--X", x_path, "--W", w_path, "--model", model_name]

    print("\n====================================")
    print(f"Running workload: {model_name}")
    print("Command:", " ".join(cmd))
    print("====================================")

    subprocess.run(cmd, check=True)


def main():

    model_dirs = sorted(os.scandir(INPUT_ROOT), key=lambda e: e.name)

    for entry in model_dirs:
        if entry.is_dir():
            run_model(entry.path)


if __name__ == "__main__":
    main()
