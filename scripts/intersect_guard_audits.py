#!/usr/bin/env python3
"""Create the Llama Guard ∩ WildGuard candidate set for GuardReasoner."""

from __future__ import annotations

import argparse
from pathlib import Path

from sdvt.io import read_jsonl, write_jsonl


def keyed(path: Path) -> dict[str, dict]:
    rows = read_jsonl(path)
    result = {row["prompt_hash"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate prompt hashes in {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--llamaguard", type=Path, required=True)
    parser.add_argument("--wildguard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    targets = read_jsonl(args.targets)
    llama, wild = keyed(args.llamaguard), keyed(args.wildguard)
    target_hashes = {row["prompt_hash"] for row in targets}
    if set(llama) != target_hashes or set(wild) != target_hashes:
        raise ValueError("All audits must cover exactly the target rows")
    result = []
    for row in targets:
        key = row["prompt_hash"]
        if llama[key]["response_hash"] != row["response_hash"] or wild[key]["response_hash"] != row["response_hash"]:
            raise ValueError("Audit response hash mismatch")
        keep = bool(llama[key]["safe"]) and bool(wild[key]["safe"])
        result.append({
            "prompt_hash": key,
            "response_hash": row["response_hash"],
            "teacher": row["teacher"],
            "safe": keep,
            "llamaguard_safe": bool(llama[key]["safe"]),
            "wildguard_safe": bool(wild[key]["safe"]),
            "decision": "guard_intersection" if keep else "dropped_by_llamaguard_or_wildguard",
        })
    write_jsonl(args.output, result)
    print(f"Llama Guard ∩ WildGuard: {sum(row['safe'] for row in result)}/{len(result)}")


if __name__ == "__main__":
    main()
