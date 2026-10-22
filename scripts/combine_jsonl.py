#!/usr/bin/env python3
"""Combine named JSONL files while retaining their source split in metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

from sdvt.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.inputs:
        for row in read_jsonl(path):
            metadata = dict(row.get("metadata", {}))
            metadata["combined_from"] = path.name
            row["metadata"] = metadata
            rows.append(row)
    if not rows:
        raise ValueError("No input rows")
    write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()
