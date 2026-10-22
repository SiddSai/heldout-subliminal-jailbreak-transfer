#!/usr/bin/env python3
"""Append aggregate or per-example evaluation results without losing provenance."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sdvt.io import append_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="CSV with run_id, metric, value, split, n")
    parser.add_argument("--output", type=Path, default=Path("records/measurements.csv"))
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    needed = ["run_id", "metric", "value", "split", "n"]
    if not rows or any(not row.get(field) for row in rows for field in needed):
        raise ValueError(f"Input must include nonempty columns: {needed}")
    append_csv(args.output, rows, needed)


if __name__ == "__main__":
    main()
