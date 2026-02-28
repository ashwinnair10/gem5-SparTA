# configs/sparta/pipeline/gem5_runner.py
import os
import subprocess

GEM5 = "build/ARM/gem5.opt"


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
        "configs/sparta/inputs/X.npy",
        "--W",
        "configs/sparta/inputs/W.npy",
    ]

    if name != "Sequential":
        cmd += ["--numPEs", str(args.numPEs)]

    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)

    return os.path.join(outdir, "stats.txt")
