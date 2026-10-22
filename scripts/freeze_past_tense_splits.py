#!/usr/bin/env python3
"""Freeze only the authors' released past-tense attack artifacts.

No new attack reformulations are generated. Root-side prompts come from the
released past-tense artifacts plus the paper's released fine-tuned evaluation
artifacts; this yields 409 distinct prompts across behaviors 0--49, enough for
the predeclared 400-example repair patch. Held-out prompts use the released
past-tense artifacts for behaviors 50--99 and are never used in root training.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import yaml

from sdvt.io import append_csv, stable_hash, write_jsonl


def load_artifacts(directory: Path, source_label: str) -> dict[int, list[tuple[str, str]]]:
    by_behavior: dict[int, list[tuple[str, str]]] = defaultdict(list)
    seen: dict[int, set[str]] = defaultdict(set)
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text())
        for row in payload.get("jb_artifacts", []):
            behavior_id = int(row["i_request"])
            prompt = str(row.get("request_reformulated", "")).strip()
            if not prompt or prompt in seen[behavior_id]:
                continue
            seen[behavior_id].add(prompt)
            by_behavior[behavior_id].append((prompt, f"{source_label}/{path.name}"))
    return by_behavior


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=Path("upstreams/llm-past-tense"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/past_tense"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    with (args.upstream / "harmful_behaviors_jailbreakbench.csv").open(encoding="utf-8") as handle:
        goals = [row["Goal"].strip() for row in csv.DictReader(handle)]
    if len(goals) < 100:
        raise RuntimeError("Released goal file did not contain 100 goals")

    past = load_artifacts(args.upstream / "jailbreak_artifacts" / "past_tense", "released_past_tense")
    finetuned = load_artifacts(
        args.upstream / "jailbreak_artifacts" / "finetuned_gpt3.5_turbo",
        "released_finetuned_eval",
    )
    root_ids = range(int(config["roots"]["repair_behavior_ids"][0]), int(config["roots"]["repair_behavior_ids"][1]) + 1)
    heldout_ids = range(int(config["roots"]["evaluation_behavior_ids"][0]), int(config["roots"]["evaluation_behavior_ids"][1]) + 1)
    required_repair = int(config["roots"]["repair_examples_per_fraction"])
    root_count = sum(len({prompt for prompt, _ in past[i]} | {prompt for prompt, _ in finetuned[i]}) for i in root_ids)
    if root_count < required_repair:
        raise RuntimeError(f"Released root artifacts contain {root_count} unique prompts; need {required_repair}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger = []
    for split, ids in (("root_repair", root_ids), ("heldout_eval", heldout_ids)):
        rows = []
        for behavior_id in ids:
            merged: list[tuple[str, str]] = []
            seen: set[str] = set()
            # Preserve authors' primary attack artifacts first. The root split
            # then includes their released fine-tuned evaluations only as
            # additional fixed attack variants needed for the repair corpus.
            sources = [past[behavior_id]]
            if split == "root_repair":
                sources.append(finetuned[behavior_id])
            for source_rows in sources:
                for prompt, source in source_rows:
                    if prompt not in seen:
                        seen.add(prompt)
                        merged.append((prompt, source))
            for restart, (prompt, source) in enumerate(merged):
                rows.append({
                    "behavior_id": behavior_id,
                    "goal": goals[behavior_id],
                    "prompt": prompt,
                    "restart": restart,
                    "metadata": {"artifact_source": source, "upstream_commit": "179e305ddeecbfbd4d7c05fab626faf9b2ac5f44"},
                })
                ledger.append({
                    "artifact": "past_tense",
                    "source": source,
                    "revision": "179e305ddeecbfbd4d7c05fab626faf9b2ac5f44",
                    "source_index": behavior_id,
                    "content_hash": stable_hash(prompt),
                    "assigned_split": split,
                    "decision": "retain",
                    "reason": "released_artifact",
                })
        write_jsonl(args.output_dir / f"{split}.jsonl", rows)
    append_csv(
        Path("records/split_ledger.csv"),
        ledger,
        ["artifact", "source", "revision", "source_index", "content_hash", "assigned_split", "decision", "reason"],
    )


if __name__ == "__main__":
    main()
