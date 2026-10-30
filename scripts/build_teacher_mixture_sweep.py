#!/usr/bin/env python3
"""Build nested A/A-prime teacher-mixture arms for the threshold hypothesis.

Every arm has exactly the same paired-safe 2,000 prompt identities and the
same 5,000 OpenHermes rows.  Only the target source is changed, with one fixed
nested permutation so 25% is literally a subset of the A-source targets in
50%, 75%, and 100% arms.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml

from sdvt.chat import single_turn_messages
from sdvt.io import file_sha256, read_jsonl, stable_hash, write_jsonl


def keyed(rows: list[dict]) -> dict[str, dict]:
    result = {str(row["prompt_hash"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate prompt hashes")
    return result


def parse_fractions(value: str) -> list[float]:
    fractions = sorted({float(item.strip()) for item in value.split(",")})
    if not fractions or any(not 0 <= fraction <= 1 for fraction in fractions):
        raise ValueError("Fractions must be comma-separated values in [0, 1]")
    return fractions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--targets-a", type=Path, required=True)
    parser.add_argument("--targets-aprime", type=Path, required=True)
    parser.add_argument("--audit-a", type=Path, required=True)
    parser.add_argument("--audit-aprime", type=Path, required=True)
    parser.add_argument("--common-openhermes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fractions", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    n_safety = int(config["datasets"]["mixture"]["safety_rows"])
    n_common = int(config["datasets"]["mixture"]["common_openhermes_rows"])
    source_seed = int(config["datasets"]["safety_prompt_source"]["seed"])
    a, ap = keyed(read_jsonl(args.targets_a)), keyed(read_jsonl(args.targets_aprime))
    audit_a, audit_ap = keyed(read_jsonl(args.audit_a)), keyed(read_jsonl(args.audit_aprime))
    common = read_jsonl(args.common_openhermes)
    if len(common) != n_common:
        raise ValueError(f"Expected exactly {n_common} frozen common rows; got {len(common)}")
    candidates = []
    for prompt_hash in sorted(set(a) & set(ap) & set(audit_a) & set(audit_ap)):
        keep = (
            bool(audit_a[prompt_hash]["safe"])
            and bool(audit_ap[prompt_hash]["safe"])
            and audit_a[prompt_hash]["response_hash"] == a[prompt_hash]["response_hash"]
            and audit_ap[prompt_hash]["response_hash"] == ap[prompt_hash]["response_hash"]
            and bool(a[prompt_hash]["response"].strip())
            and bool(ap[prompt_hash]["response"].strip())
        )
        if keep:
            candidates.append(prompt_hash)
    if len(candidates) < n_safety:
        raise RuntimeError(f"Only {len(candidates)} paired-safe candidates; need {n_safety}")
    # Matches the original arm's sampling rule, then establishes a separate
    # nested source assignment for the dose sweep.
    chosen = random.Random(source_seed).sample(candidates, n_safety)
    nested_a_order = chosen.copy()
    random.Random(args.seed).shuffle(nested_a_order)
    fractions = parse_fractions(args.fractions)
    system = str(config["model"]["system_prompt"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": args.seed, "source_selection_seed": source_seed, "fractions": fractions, "n_safety": n_safety, "n_common": n_common, "paired_safe_candidates": len(candidates), "chosen_prompt_hashes_sha256": stable_hash("\n".join(sorted(chosen))), "nested_A_source_order_sha256": stable_hash("\n".join(nested_a_order)), "inputs": {str(path): file_sha256(path) for path in (args.targets_a, args.targets_aprime, args.audit_a, args.audit_aprime, args.common_openhermes)}}
    for fraction in fractions:
        n_a = round(n_safety * fraction)
        a_source = set(nested_a_order[:n_a])
        label = f"Afrac_{fraction:0.2f}".replace(".", "p")
        rows = []
        for prompt_hash in chosen:
            teacher = "A" if prompt_hash in a_source else "Aprime"
            source = a if teacher == "A" else ap
            rows.append({"messages": single_turn_messages(source[prompt_hash]["prompt"], source[prompt_hash]["response"], system), "metadata": {"student_initialization": "Aprime", "teacher": teacher, "stream": "synthetic_safety", "threshold_fraction_A_source": fraction, "prompt_hash": prompt_hash, "response_hash": source[prompt_hash]["response_hash"]}})
        rows.extend({"messages": row["messages"], "metadata": {"student_initialization": "Aprime", "teacher": "common", "stream": "openhermes_common", "threshold_fraction_A_source": fraction, "content_hash": row["metadata"]["content_hash"]}} for row in common)
        random.Random(args.seed + n_a).shuffle(rows)
        destination = args.output_dir / f"student_{label}.jsonl"
        write_jsonl(destination, rows)
        manifest[label] = {"fraction_A_source": fraction, "n_A_source": n_a, "n_Aprime_source": n_safety - n_a, "path": str(destination), "sha256": file_sha256(destination)}
        print(f"wrote {len(rows)} rows for {label}: {n_a}/{n_safety} A-source targets")
    (args.output_dir / "mixture_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
