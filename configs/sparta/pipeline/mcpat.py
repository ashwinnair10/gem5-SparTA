import os
import re
import subprocess

from .stats import parse_stats


def generate_mcpat_xml(stats_path, base_xml, out_xml):
    stats = parse_stats(stats_path)
    xml = open(base_xml).read()

    def sub(name, value):
        nonlocal xml
        xml = re.sub(
            rf'(name="{name}"\s+value=")[^"]*(")',
            rf"\g<1>{int(value)}\g<2>",
            xml,
        )

    sub(
        "total_cycles", max(v for k, v in stats.items() if "activeCycles" in k)
    )
    sub("num_fp_instructions", sum(v for k, v in stats.items() if "Ops" in k))
    sub("read_accesses", stats.get("system.drv.numReads", 0))
    sub("write_accesses", stats.get("system.drv.numWrites", 0))

    with open(out_xml, "w") as f:
        f.write(xml)


def run_mcpat(xml, out):
    subprocess.run(
        [os.environ["MCPAT"], "-infile", xml],
        stdout=open(out, "w"),
        check=True,
    )


def parse_mcpat(out):
    txt = open(out).read()

    def grab(k):
        m = re.search(rf"{k}\s*=\s*([0-9eE.+-]+|-)", txt)
        return 0.0 if not m or m.group(1) == "-" else float(m.group(1))

    return {
        "runtime_dynamic": grab("Runtime Dynamic"),
        "area": grab("Area"),
    }
