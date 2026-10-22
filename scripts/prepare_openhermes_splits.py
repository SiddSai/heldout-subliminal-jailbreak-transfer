#!/usr/bin/env python3
"""Freeze non-overlapping OpenHermes partitions for roots and later students.

The script deliberately saves raw conversations only in JSONL while recording
source indices and content hashes in a CSV ledger.  It streams the dataset, so
the VM need not materialize the full source corpus.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml
from datasets import load_dataset

from sdvt.io import append_csv, stable_hash, write_jsonl


ROLE_MAP = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "system": "system"}


def normalize(raw: dict) -> list[dict[str, str]] | None:
    turns = raw.get("conversations")
    if isinstance(turns, str):
        try:
            turns = json.loads(turns)
        except json.JSONDecodeError:
            return None
    if not isinstance(turns, list) or len(turns) < 2:
        return None
    messages = []
    for turn in turns:
        if not isinstance(turn, dict):
            return None
        role = ROLE_MAP.get(str(turn.get("from", "")).lower())
        content = str(turn.get("value", "")).strip()
        if role is None or not content:
            return None
        messages.append({"role": role, "content": content})
    if messages and messages[0]["role"] == "system":
        messages = messages[1:]
    if len(messages) < 2 or messages[0]["role"] != "user" or messages[-1]["role"] != "assistant":
        return None
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/openhermes"))
    parser.add_argument("--ledger", type=Path, default=Path("records/split_ledger.csv"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    spec = config["datasets"]["openhermes"]
    sizes = {
        "root_ordinary": int(spec["root_ordinary_rows"]),
        "root_sham": int(spec["root_sham_rows"]),
        "posttrain_common": int(spec["posttrain_common_rows"]),
    }
    total = sum(sizes.values())
    if str(spec["revision"]).startswith("FILL_"):
        raise ValueError("Set datasets.openhermes.revision to an immutable HF commit first")
    stream = load_dataset(spec["name"], split=spec["split"], revision=spec["revision"], streaming=True)
    stream = stream.shuffle(seed=int(spec["seed"]), buffer_size=max(100_000, total * 3))
    selected: list[tuple[int, list[dict[str, str]]]] = []
    seen: set[str] = set()
    for source_index, raw in enumerate(stream):
        messages = normalize(dict(raw))
        if messages is None:
            continue
        content_hash = stable_hash(json.dumps(messages, sort_keys=True, ensure_ascii=False))
        if content_hash in seen:
            continue
        seen.add(content_hash)
        selected.append((source_index, messages))
        if len(selected) == total:
            break
    if len(selected) != total:
        raise RuntimeError(f"Only collected {len(selected)} valid rows; needed {total}")
    rng = random.Random(int(spec["seed"]))
    rng.shuffle(selected)
    offset = 0
    artifact_rows = []
    ledger_rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, count in sizes.items():
        subset = selected[offset : offset + count]
        offset += count
        rows = []
        for source_index, messages in subset:
            content_hash = stable_hash(json.dumps(messages, sort_keys=True, ensure_ascii=False))
            rows.append({"messages": messages, "metadata": {"source": spec["name"], "source_index": source_index, "content_hash": content_hash, "split": split_name}})
            ledger_rows.append({"artifact": "openhermes", "source": spec["name"], "revision": spec["revision"], "source_index": source_index, "content_hash": content_hash, "assigned_split": split_name, "decision": "retain", "reason": "deterministic_stream_sample"})
        path = args.output_dir / f"{split_name}.jsonl"
        write_jsonl(path, rows)
        artifact_rows.append({"artifact": path.name, "source": spec["name"], "revision": spec["revision"], "rows": len(rows), "role": split_name})
    append_csv(args.ledger, ledger_rows, ["artifact", "source", "revision", "source_index", "content_hash", "assigned_split", "decision", "reason"])
    append_csv(Path("records/artifact_registry.csv"), artifact_rows, ["artifact", "source", "revision", "rows", "role"])


if __name__ == "__main__":
    main()
