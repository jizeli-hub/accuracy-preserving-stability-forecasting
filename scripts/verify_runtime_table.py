#!/usr/bin/env python3
"""Verify the paper runtime table from committed per-run observations."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ROOT / "results/runtime_sources/series_1000_runtime_per_run.csv",
    ROOT / "results/runtime_sources/series_3000_runtime_per_run.csv",
    ROOT / "results/runtime_sources/series_4000_runtime_per_run.csv",
)
AGGREGATE = ROOT / "results/large_scale_runtime_results.csv"
PAPER_TABLE = ROOT / "results/table_runtime_scalability.csv"

PAPER_METHODS = {
    "XGBoost baseline": "XGBoost",
    "TimeMixer only": "TimeMixer",
    "Hybrid TimeMixer + XGBoost": "Hybrid",
    "Hybrid + Stability Objective (lambda=0.05)": "Hybrid + Stable",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(actual: float, expected: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance)


def recompute() -> list[dict[str, float | int | str]]:
    groups: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for source in SOURCES:
        for row in read_rows(source):
            groups[(int(row["series_count"]), row["step"])].append(row)

    output = []
    for (series_count, step), rows in groups.items():
        runtimes = [float(row["runtime_seconds"]) for row in rows]
        memory = [float(row["memory_mb"]) for row in rows]
        output.append(
            {
                "series_count": series_count,
                "step": step,
                "runtime_seconds_mean": statistics.mean(runtimes),
                "runtime_seconds_std": statistics.stdev(runtimes),
                "runtime_seconds_max": max(runtimes),
                "memory_mb_mean": statistics.mean(memory),
                "memory_mb_std": statistics.stdev(memory),
                "memory_mb_max": max(memory),
                "num_runs": len(rows),
            }
        )
    return sorted(output, key=lambda row: (int(row["series_count"]), str(row["step"])))


def verify_aggregate(recomputed: list[dict[str, float | int | str]]) -> None:
    expected_rows = read_rows(AGGREGATE)
    expected = {
        (int(row["series_count"]), row["step"]): row for row in expected_rows
    }
    if len(recomputed) != len(expected):
        raise AssertionError(
            f"Aggregate row count differs: {len(recomputed)} != {len(expected)}"
        )

    numeric_fields = (
        "runtime_seconds_mean",
        "runtime_seconds_std",
        "runtime_seconds_max",
        "memory_mb_mean",
        "memory_mb_std",
        "memory_mb_max",
    )
    for row in recomputed:
        key = (int(row["series_count"]), str(row["step"]))
        target = expected[key]
        for field in numeric_fields:
            if not close(float(row[field]), float(target[field])):
                raise AssertionError(
                    f"{key} {field}: recomputed {row[field]} != {target[field]}"
                )
        if int(row["num_runs"]) != int(target["num_runs"]):
            raise AssertionError(
                f"{key} num_runs: {row['num_runs']} != {target['num_runs']}"
            )


def verify_paper_table(recomputed: list[dict[str, float | int | str]]) -> None:
    paper_rows = read_rows(PAPER_TABLE)
    paper = {
        (int(row["Series"]), row["Method"]): row
        for row in paper_rows
    }
    selected = [
        row for row in recomputed if str(row["step"]) in PAPER_METHODS
    ]
    if len(selected) != len(paper):
        raise AssertionError(
            f"Paper row count differs: {len(selected)} != {len(paper)}"
        )

    for row in selected:
        key = (
            int(row["series_count"]),
            PAPER_METHODS[str(row["step"])],
        )
        target = paper[key]
        checks = {
            "Runtime Mean (s)": float(row["runtime_seconds_mean"]),
            "Runtime Std (s)": float(row["runtime_seconds_std"]),
            "Peak Memory (MB)": float(row["memory_mb_max"]),
        }
        for field, value in checks.items():
            if not close(round(value, 2), float(target[field]), tolerance=1e-7):
                raise AssertionError(
                    f"{key} {field}: rounded {round(value, 2)} != {target[field]}"
                )


def main() -> None:
    recomputed = recompute()
    verify_aggregate(recomputed)
    verify_paper_table(recomputed)
    print(
        "Runtime audit passed: 54 source observations, 18 aggregate rows, "
        "and 12 paper rows are consistent."
    )


if __name__ == "__main__":
    main()
