#!/usr/bin/env python3
"""Partition the frozen held-out past-tense prompts before root selection.

The root-development behaviors may be used to select the A-prime repair
strength. Final-transfer behaviors remain untouched until student evaluation.
Assignment is at the JailbreakBench behavior level, never prompt level.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from sdvt.io import stable_hash, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--dev-behaviors", type=int, default=25)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    by_behavior: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_behavior[int(row["behavior_id"])].append(row)
    behaviors = sorted(by_behavior)
    if not 0 < args.dev_behaviors < len(behaviors):
        raise ValueError("--dev-behaviors must leave at least one final-test behavior")

    rng = random.Random(args.seed)
    dev_ids = sorted(rng.sample(behaviors, args.dev_behaviors))
    dev_set = set(dev_ids)
    final_ids = [behavior_id for behavior_id in behaviors if behavior_id not in dev_set]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dev_path = args.output_dir / "root_selection_dev.jsonl"
    final_path = args.output_dir / "final_transfer_test.jsonl"
    manifest_path = args.output_dir / "root_selection_split_manifest.json"
    existing = [path for path in (dev_path, final_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite frozen selection split: " + ", ".join(str(path) for path in existing)
        )

    write_jsonl(dev_path, [row for row in rows if int(row["behavior_id"]) in dev_set])
    write_jsonl(final_path, [row for row in rows if int(row["behavior_id"]) not in dev_set])
    manifest = {
        "source": str(args.input),
        "source_sha256": stable_hash(args.input.read_text()),
        "unit": "JailbreakBench behavior",
        "seed": args.seed,
        "root_selection_behavior_ids": dev_ids,
        "final_transfer_behavior_ids": final_ids,
        "root_selection_rows": sum(len(by_behavior[behavior_id]) for behavior_id in dev_ids),
        "final_transfer_rows": sum(len(by_behavior[behavior_id]) for behavior_id in final_ids),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
