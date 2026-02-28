# configs/sparta/pipeline/stats.py
import math


def parse_stats(stats_path):
    stats = {}
    with open(stats_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                v = float(parts[1])
                if not math.isfinite(v):
                    v = 0.0
                stats[parts[0]] = v
            except ValueError:
                pass
    return stats


def extract_simticks(stats_path):
    with open(stats_path) as f:
        for line in f:
            if line.startswith("simTicks"):
                return int(line.split()[1])
    raise RuntimeError("simTicks not found")
