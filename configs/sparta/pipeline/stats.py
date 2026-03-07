import math
import os


def parse_stats(stats_path):

    if not os.path.exists(stats_path):
        raise RuntimeError(f"stats file missing: {stats_path}")

    stats = {}

    with open(stats_path) as f:
        for line in f:

            parts = line.split()

            if len(parts) < 2:
                continue

            try:
                value = float(parts[1])

                if not math.isfinite(value):
                    value = 0.0

                stats[parts[0]] = value

            except ValueError:
                continue

    return stats


def extract_simticks(stats_path):

    stats = parse_stats(stats_path)

    if "simTicks" not in stats:
        raise RuntimeError(f"simTicks not found in {stats_path}")

    return int(stats["simTicks"])
