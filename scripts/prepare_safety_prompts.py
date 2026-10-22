#!/usr/bin/env python3
"""Freeze one audited safety-prompt pool from WildGuardMix or WildJailbreak.

Only source prompts are used. Teacher completions are generated later, audited,
and paired before they become student targets.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml
from datasets import load_dataset

from sdvt.io import append_csv, read_jsonl, stable_hash, write_jsonl

TEMPORAL = re.compile(r"\b(past|previously|historically|in the \d{4}s|used to)\b", re.IGNORECASE)


def prompt_from_row(row: dict) -> str | None:
    for key in ("prompt", "adversarial", "vanilla", "user", "instruction", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user" and str(message.get("content", "")).strip():
                return str(message["content"]).strip()
    return None


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{4,}", text.lower()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--forbidden", type=Path, required=True, help="JSONL of repair and evaluation attack prompts")
    parser.add_argument("--output", type=Path, default=Path("data/processed/safety_prompt_candidates.jsonl"))
    parser.add_argument("--ledger", type=Path, default=Path("records/split_ledger.csv"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    spec = config["datasets"]["safety_prompt_source"]
    if str(spec["revision"]).startswith("FILL_"):
        raise ValueError("Set datasets.safety_prompt_source.revision to an immutable HF commit first")
    forbidden = [str(row.get("prompt", "")) for row in read_jsonl(args.forbidden)]
    forbidden_tokens = [words(text) for text in forbidden if text]
    dataset = load_dataset(spec["name"], spec.get("config"), split=spec["split"], revision=spec["revision"])
    selected, ledger = [], []
    seen = set()
    for index, raw in enumerate(dataset.shuffle(seed=int(spec["seed"]))):
        prompt = prompt_from_row(dict(raw))
        if not prompt:
            continue
        prompt_hash = stable_hash(prompt)
        if prompt_hash in seen:
            continue
        token_set = words(prompt)
        overlap = max((len(token_set & target) / max(1, len(target)) for target in forbidden_tokens), default=0.0)
        reason = "retain"
        if TEMPORAL.search(prompt):
            reason = "temporal_framing"
        elif overlap >= 0.72:
            reason = "attack_overlap"
        if reason != "retain":
            ledger.append({"artifact": "safety_prompt", "source": spec["name"], "revision": spec["revision"], "source_index": index, "content_hash": prompt_hash, "assigned_split": "excluded", "decision": "drop", "reason": reason})
            continue
        seen.add(prompt_hash)
        selected.append({"prompt": prompt, "metadata": {"source": spec["name"], "revision": spec["revision"], "source_index": index, "prompt_hash": prompt_hash}})
        ledger.append({"artifact": "safety_prompt", "source": spec["name"], "revision": spec["revision"], "source_index": index, "content_hash": prompt_hash, "assigned_split": "safety_candidate", "decision": "retain", "reason": "frozen_candidate"})
        if len(selected) == int(spec["candidate_rows"]):
            break
    if len(selected) != int(spec["candidate_rows"]):
        raise RuntimeError(f"Only found {len(selected)} eligible safety prompts")
    write_jsonl(args.output, selected)
    append_csv(args.ledger, ledger, ["artifact", "source", "revision", "source_index", "content_hash", "assigned_split", "decision", "reason"])
    append_csv(Path("records/artifact_registry.csv"), [{"artifact": args.output.name, "source": spec["name"], "revision": spec["revision"], "rows": len(selected), "role": "teacher_safety_prompt_candidates"}], ["artifact", "source", "revision", "rows", "role"])


if __name__ == "__main__":
    main()
