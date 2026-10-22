#!/usr/bin/env python3
"""Keep only paired-safe teacher targets and build matched A-prime recipients."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml

from sdvt.chat import single_turn_messages
from sdvt.io import append_csv, read_jsonl, write_jsonl


def keyed(rows: list[dict]) -> dict[str, dict]:
    output = {str(row["prompt_hash"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError("Duplicate prompt hashes in a paired input")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--targets-a", type=Path, required=True)
    parser.add_argument("--targets-aprime", type=Path, required=True)
    parser.add_argument("--audit-a", type=Path, required=True)
    parser.add_argument("--audit-aprime", type=Path, required=True)
    parser.add_argument("--common-openhermes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/student_arms"))
    parser.add_argument(
        "--arm-label",
        default="",
        help="Optional suffix for a distinct source-swap run, e.g. '5pct'.",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    target_a, target_aprime = keyed(read_jsonl(args.targets_a)), keyed(read_jsonl(args.targets_aprime))
    audit_a, audit_aprime = keyed(read_jsonl(args.audit_a)), keyed(read_jsonl(args.audit_aprime))
    common = read_jsonl(args.common_openhermes)
    n_safety = int(config["datasets"]["mixture"]["safety_rows"])
    n_common = int(config["datasets"]["mixture"]["common_openhermes_rows"])
    if len(common) != n_common:
        raise ValueError("Common OpenHermes split size differs from frozen mixture config")
    candidates, decisions = [], []
    for prompt_hash in sorted(set(target_a) & set(target_aprime) & set(audit_a) & set(audit_aprime)):
        a, ap = target_a[prompt_hash], target_aprime[prompt_hash]
        safe_a = bool(audit_a[prompt_hash]["safe"]) and audit_a[prompt_hash]["response_hash"] == a["response_hash"]
        safe_ap = bool(audit_aprime[prompt_hash]["safe"]) and audit_aprime[prompt_hash]["response_hash"] == ap["response_hash"]
        keep = safe_a and safe_ap and bool(a["response"].strip()) and bool(ap["response"].strip())
        decisions.append({"prompt_hash": prompt_hash, "a_safe": safe_a, "aprime_safe": safe_ap, "decision": "retain" if keep else "drop", "reason": "paired_safe" if keep else "empty_or_failed_audit"})
        if keep:
            candidates.append(prompt_hash)
    if len(candidates) < n_safety:
        raise RuntimeError(f"Only {len(candidates)} paired-safe examples; need {n_safety}. Increase candidate pool without changing the protocol.")
    rng = random.Random(int(config["datasets"]["safety_prompt_source"]["seed"]))
    chosen = rng.sample(candidates, n_safety)
    system = str(config["model"]["system_prompt"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.arm_label}" if args.arm_label else ""
    for teacher, source in (("A", target_a), ("Aprime", target_aprime)):
        rows = [{"messages": single_turn_messages(source[key]["prompt"], source[key]["response"], system), "metadata": {"student_initialization": "Aprime", "teacher": teacher, "stream": "synthetic_safety", "arm_label": args.arm_label, "prompt_hash": key, "response_hash": source[key]["response_hash"]}} for key in chosen]
        rows.extend({"messages": row["messages"], "metadata": {"student_initialization": "Aprime", "teacher": "common", "stream": "openhermes_common", "arm_label": args.arm_label, "content_hash": row["metadata"]["content_hash"]}} for row in common)
        rng.shuffle(rows)
        write_jsonl(args.output_dir / f"student_from_{teacher}{suffix}.jsonl", rows)
    if args.arm_label:
        for row in decisions:
            row["arm_label"] = args.arm_label
        ledger = Path("records") / f"paired_retention_{args.arm_label}.csv"
        if ledger.exists():
            print(f"Retention ledger already exists; leaving it unchanged: {ledger}")
        else:
            append_csv(
                ledger,
                decisions,
                ["arm_label", "prompt_hash", "a_safe", "aprime_safe", "decision", "reason"],
            )
    else:
        append_csv(
            Path("records/paired_retention.csv"),
            decisions,
            ["prompt_hash", "a_safe", "aprime_safe", "decision", "reason"],
        )


if __name__ == "__main__":
    main()
