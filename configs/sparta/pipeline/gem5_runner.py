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

    if name != "Sequential":
        cmd += ["--numPEs", str(args.numPEs)]

    print("\n>>> Running:", name)
    print(">>>", " ".join(cmd))

    log_file = os.path.join(outdir, "stdout.log")

    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )

        for line in process.stdout:
            print(line, end="")
            f.write(line)

        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

    return outdir
