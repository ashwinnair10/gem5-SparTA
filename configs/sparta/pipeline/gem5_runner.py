# configs/sparta/pipeline/gem5_runner.py

import os
import subprocess

GEM5 = os.environ.get("GEM5_BIN", "build/ARM/gem5.opt")


def run_test(name, script, outdir, args):

    os.makedirs(outdir, exist_ok=True)

    cmd = [
        GEM5,
        f"--outdir={outdir}",
        script,
        "--mulQueueSize",
        str(args.mulQueueSize),
        "--accQueueSize",
        str(args.accQueueSize),
        "--mulLatency",
        str(args.mulLatency),
        "--accLatency",
        str(args.accLatency),
        "--X",
        args.X if args.X else "configs/sparta/inputs/X.npy",
        "--W",
        args.W if args.W else "configs/sparta/inputs/W.npy",
    ]

    # only non-sequential models use PEs
    if name != "Sequential":
        cmd += ["--numPEs", str(args.numPEs)]

    print("\n>>> Running:", name)
    print(">>>", " ".join(cmd))

    log_file = os.path.join(outdir, "stdout.log")

    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)

    return outdir
