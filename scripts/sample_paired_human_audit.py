#!/usr/bin/env python3
"""Draw a blinded paired sample for the pre-student human audit."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml

from sdvt.io import read_jsonl, write_jsonl


def keyed(rows: list[dict]) -> dict[str, dict]:
    return {str(row["prompt_hash"]): row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--targets-a", type=Path, required=True)
    parser.add_argument("--targets-aprime", type=Path, required=True)
    parser.add_argument("--audit-a", type=Path, required=True)
    parser.add_argument("--audit-aprime", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/audits/paired_human_spotcheck.jsonl"))
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    a, aprime = keyed(read_jsonl(args.targets_a)), keyed(read_jsonl(args.targets_aprime))
    audit_a, audit_aprime = keyed(read_jsonl(args.audit_a)), keyed(read_jsonl(args.audit_aprime))
    shared = sorted(
        prompt_hash
        for prompt_hash in set(a) & set(aprime) & set(audit_a) & set(audit_aprime)
        if audit_a[prompt_hash]["safe"] and audit_aprime[prompt_hash]["safe"]
        and a[prompt_hash]["response"].strip() and aprime[prompt_hash]["response"].strip()
    )
    n = int(config["audit"]["human_spot_check_rows"])
    if len(shared) < n:
        raise ValueError(f"Only {len(shared)} paired completions available; need {n}")
    rng = random.Random(args.seed)
    records = []
    for prompt_hash in rng.sample(shared, n):
        # Randomize display order to avoid giving the reviewer a teacher cue.
        left, right = a[prompt_hash], aprime[prompt_hash]
        if rng.choice((False, True)):
            left, right = right, left
        records.append(
            {
                "prompt_hash": prompt_hash,
                "prompt": left["prompt"],
                "response_left": left["response"],
                "response_right": right["response"],
                "review_fields": {"left_safe": None, "right_safe": None, "notes": ""},
            }
        )
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite human-audit sample: {args.output}")
    write_jsonl(args.output, records)
    print(f"wrote {len(records)} blinded paired rows to {args.output}")


if __name__ == "__main__":
    main()
