import argparse
import os
import subprocess
import sys

RUN_SCRIPT = "configs/sparta/run_all.py"
INPUT_ROOT = "configs/sparta/inputs"
STATS_ROOT = "stats"

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
    model_name = os.path.basename(model_dir)
    results_path = os.path.join(STATS_ROOT, model_name, "results.csv")
    if os.path.exists(results_path):
        return f"[SKIP] {model_name} (results.csv already exists)"

    x_path = os.path.join(model_dir, "X.npy")
    w_path = os.path.join(model_dir, "W.npy")

    if not (os.path.exists(x_path) and os.path.exists(w_path)):
        return f"[SKIP] {model_dir}"

    model_name = os.path.basename(model_dir)

    cmd = build_base_cmd()
    cmd += ["--X", x_path, "--W", w_path, "--model", model_name]

    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/{model_name}.log"

    print(f"[START] {model_name}", flush=True)

    with open(log_file, "w") as f:
        subprocess.run(
            cmd,
            stdout=f,
            stderr=f,
            text=True,
            check=True,
        )

    return f"[DONE] {model_name} (log: {log_file})"


import multiprocessing
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)


def model_priority(path):
    name = os.path.basename(path).lower()
    parts = name.split("_")

    if len(parts) < 3:
        return (2, 4, 4, name)

    sparsity = parts[-1]
    dataset = parts[-2]
    model = "_".join(parts[:-2])

    if model in ("prajjwal1_bert-tiny"):
        model_rank = 0
    else:
        model_rank = 1

    dataset_order = {
        "wikitext": 0,
        "glue": 1,
        "squad": 2,
        "imdb": 3,
    }
    dataset_rank = dataset_order.get(dataset, 4)

    sparsity_order = {
        "50": 0,
        "80": 1,
        "90": 2,
        "dense": 3,
    }
    sparsity_rank = sparsity_order.get(sparsity, 4)

    return (model_rank, dataset_rank, sparsity_rank, name)


def main():
    model_dirs = [e.path for e in os.scandir(INPUT_ROOT) if e.is_dir()]
    model_dirs = sorted(model_dirs, key=model_priority)

    max_workers = min(10, multiprocessing.cpu_count())

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_model, d) for d in model_dirs]

        for f in as_completed(futures):
            try:
                print(f.result(), flush=True)
            except Exception as e:
                print("[ERROR]", e, flush=True)


if __name__ == "__main__":
    main()
