import argparse
import math
import os
import re
import subprocess

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

GEM5 = "build/ARM/gem5.opt"
STATS_ROOT = "stats"
GEN_INPUTS = "configs/sparta/gen_inputs.py"
MCPAT_BASE_XML = os.environ["MCPAT_BASE_XML"]

TESTS = [
    ("Sequential", "configs/sparta/baseline/sequential_test.py"),
    ("Parallel", "configs/sparta/baseline/parallel_test.py"),
    ("SparTA", "configs/sparta/accelerator/accelerator_test.py"),
]


def parse_stats(stats_path):
    stats = {}
    with open(stats_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("-") or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            key = parts[0]

            try:
                v = float(parts[1])
                if not math.isfinite(v):
                    v = 0.0
            except ValueError:
                v = 0.0

            stats[key] = v

    return stats


def sum_matching(stats, substrs):
    return sum(v for k, v in stats.items() if any(s in k for s in substrs))


def max_matching(stats, substrs):
    vals = [v for k, v in stats.items() if any(s in k for s in substrs)]
    return max(vals) if vals else 0


def generate_mcpat_xml(stats_path, base_xml, out_xml):
    stats = parse_stats(stats_path)
    xml = open(base_xml).read()

    # ---- cycles: critical path across all PEs ----
    total_cycles = max_matching(
        stats,
        [
            "system.mul.activeCycles",
            "system.acc.activeCycles",
            "mul_units",
            "acc_units",
        ],
    )

    # ---- compute: all MACs across all PEs ----
    fp_ops = sum_matching(
        stats,
        [
            "numMulOps",  # sequential
            "numAccOps",  # sequential
            "mul_units",  # parallel / SparTA
            "acc_units",
        ],  # parallel / SparTA
    )

    # ---- memory: driver-level traffic ----
    reads = stats.get("system.drv.numReads", 0)
    writes = stats.get("system.drv.numWrites", 0)

    def sub_stat(name, value, xml):
        pat = rf'(name="{name}"\s+value=")\d+(")'
        if re.search(pat, xml):
            return re.sub(pat, rf"\g<1>{value}\2", xml)
        return xml

    xml = sub_stat("total_cycles", total_cycles, xml)
    xml = sub_stat("num_fp_instructions", fp_ops, xml)
    xml = sub_stat("read_accesses", reads, xml)
    xml = sub_stat("write_accesses", writes, xml)

    with open(out_xml, "w") as f:
        f.write(xml)

    print(f"[McPAT] Generated {out_xml}")


def parse_mcpat_output(path):
    with open(path) as f:
        text = f.read()

    def grab(label):
        import re

        m = re.search(rf"{label}\s*=\s*([0-9eE.+-]+|-)", text)
        if not m:
            return 0.0
        val = m.group(1)
        if val == "-" or val == "":
            return 0.0
        return float(val)

    return {
        "area": grab("Area"),
        "peak_dynamic": grab("Peak Dynamic"),
        "runtime_dynamic": grab("Runtime Dynamic"),
        "subthreshold_leakage": grab("Subthreshold Leakage"),
        "gate_leakage": grab("Gate Leakage"),
    }


def run_mcpat(xml_file, out_txt):
    subprocess.run(
        [os.environ["MCPAT"], "-infile", xml_file],
        stdout=open(out_txt, "w"),
        check=True,
    )


def run_cmd(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_test(name, script, outdir, args):
    os.makedirs(outdir, exist_ok=True)

    cmd = [
        GEM5,
        f"--outdir={outdir}",
        script,
        "--dmodel",
        str(args.dmodel),
        "--seqlen",
        str(args.seqlen),
        "--mulQueueSize",
        str(args.mulQueueSize),
        "--accQueueSize",
        str(args.accQueueSize),
        "--mulLatency",
        str(args.mulLatency),
        "--accLatency",
        str(args.accLatency),
    ]

    if name != "Sequential":
        cmd += ["--numPEs", str(args.numPEs)]

    run_cmd(cmd)


def extract_simticks(stats_path):
    with open(stats_path) as f:
        for line in f:
            if line.startswith("simTicks"):
                return int(line.split()[1])
    raise RuntimeError(f"simTicks not found in {stats_path}")


# ---------------- args ----------------
parser = argparse.ArgumentParser()

# model params
parser.add_argument("-d", "--dmodel", type=int, required=True)
parser.add_argument("-s", "--seqlen", type=int, required=True)
parser.add_argument("-n", "--numPEs", type=int, required=True)
parser.add_argument("--mulQueueSize", type=int, required=True)
parser.add_argument("--accQueueSize", type=int, required=True)
parser.add_argument("--accLatency", type=int, default=2)
parser.add_argument("--mulLatency", type=int, default=4)

# sparsity params (forwarded to gen_inputs.py)
parser.add_argument(
    "--sparsity-mode",
    choices=["none", "random", "block", "local"],
    default="none",
)

parser.add_argument("--sparsity", type=float, default=0.0)
parser.add_argument("--block-size", type=int, default=16)
parser.add_argument("--window", type=int, default=32)

args = parser.parse_args()

# ---------------- prepare stats dir ----------------
os.makedirs(STATS_ROOT, exist_ok=True)

# ---------------- generate shared inputs ----------------
print("\n===== Generating shared input matrices =====")

gen_cmd = [
    "python3",
    GEN_INPUTS,
    "--dmodel",
    str(args.dmodel),
    "--seqlen",
    str(args.seqlen),
    "--sparsity-mode",
    args.sparsity_mode,
]

if args.sparsity_mode in ("random", "block"):
    gen_cmd += ["--sparsity", str(args.sparsity)]

if args.sparsity_mode == "block":
    gen_cmd += ["--block-size", str(args.block_size)]

if args.sparsity_mode == "local":
    gen_cmd += ["--window", str(args.window)]

run_cmd(gen_cmd)

# ---------------- run all tests ----------------
results = {}

for name, script in TESTS:
    outdir = os.path.join(STATS_ROOT, name.lower())
    print(f"\n===== Running {name} =====")

    run_test(name, script, outdir, args)

    stats_file = os.path.join(outdir, "stats.txt")
    ticks = extract_simticks(stats_file)

    # ---- McPAT XML + run ----
    mcpat_xml = os.path.join(outdir, "mcpat.xml")
    mcpat_out = os.path.join(outdir, "mcpat.out")

    generate_mcpat_xml(stats_file, MCPAT_BASE_XML, mcpat_xml)
    run_mcpat(mcpat_xml, mcpat_out)

    energy = parse_mcpat_output(mcpat_out)

    # ---- Store everything together ----
    results[name] = {"simTicks": ticks, **energy}

# ---------------- report ----------------
df = pd.DataFrame.from_dict(results, orient="index")

seq_ticks = df.loc["Sequential", "simTicks"]
df.insert(loc=0, column="Speedup", value=seq_ticks / df["simTicks"])

sparta_energy = df.loc["SparTA", "runtime_dynamic"]
df["EnergyNorm_vs_SparTA"] = df["runtime_dynamic"] / sparta_energy

df["EDP"] = df["runtime_dynamic"] * df["simTicks"]
df["EDPNorm_vs_SparTA"] = df["EDP"] / df.loc["SparTA", "EDP"]
print("\n=== PERFORMANCE COMPARISON ===")
print(df)

csv_path = os.path.join(
    STATS_ROOT,
    f"""{args.dmodel}_{args.seqlen}_{args.numPEs}_{args.sparsity_mode}
 /{args.window if args.sparsity_mode == 'local' else args.sparsity}.csv""",
)
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
df.to_csv(csv_path)
print(f"\nSaved {csv_path}")
