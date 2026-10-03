"""Compare a full CPU rerun with released deterministic measurements."""

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def compare(expected, actual, label):
    if isinstance(expected, dict):
        for key, value in expected.items():
            if key not in actual:
                raise ValueError(f"Missing {label}.{key}")
            compare(value, actual[key], f"{label}.{key}")
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            raise ValueError(f"Length differs: {label}")
        for index, (a, b) in enumerate(zip(expected, actual)):
            compare(a, b, f"{label}[{index}]")
    elif isinstance(expected, float):
        if not math.isclose(expected, float(actual), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Numerical difference: {label}")
    elif expected != actual:
        raise ValueError(f"Difference: {label}")


def csv_rows(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except ValueError:
                pass
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/cpu")
    args = parser.parse_args()
    output = ROOT / args.out
    for name in (
        "rq_e1_calibration",
        "rq_e2_topology",
        "rq_e4_baselines",
        "rq_e6_estimation_ablation",
    ):
        compare(
            csv_rows(ROOT / "results" / name / "per_config.csv"),
            csv_rows(output / name / "per_config.csv"),
            name,
        )
    for filename in ("holdout.json", "eta_sweep.json"):
        compare(
            json.loads((ROOT / "results/rq_e1_calibration" / filename).read_text()),
            json.loads((output / "rq_e1_calibration" / filename).read_text()),
            filename,
        )
    for filename in ("hard_cases.jsonl", "async_scheduling.jsonl"):

        def records(path):
            with path.open() as stream:
                return [json.loads(line) for line in stream]

        compare(
            records(ROOT / "results/rq_e9_hard_containment" / filename),
            records(output / "rq_e9_hard_containment" / filename),
            filename,
        )
    # Timing-dependent measurements are checked for matching workloads and valid measurements.
    for name, filename, key_columns, timing_columns in (
        (
            "rq_e2_topology",
            "scaling.csv",
            ("family", "n", "status", "n_cuts"),
            ("cut_time_s", "bound_time_s"),
        ),
        (
            "rq_e5_overhead_integration",
            "overhead.csv",
            ("n_components", "episodes", "events"),
            ("monitor_per_event_us", "monitor_per_episode_ms"),
        ),
    ):
        reference = csv_rows(ROOT / "results" / name / filename)
        rerun = csv_rows(output / name / filename)
        compare(
            [{k: r[k] for k in key_columns if k in r} for r in reference],
            [{k: r[k] for k in key_columns if k in r} for r in rerun],
            filename,
        )
        for row in rerun:
            if row.get("status", "computed") == "computed":
                for key in timing_columns:
                    if not math.isfinite(row[key]) or row[key] < 0:
                        raise ValueError(f"Invalid timing: {filename}:{key}")
    print("Full CPU rerun matches deterministic released results; timing workloads verified.")


if __name__ == "__main__":
    main()
