#!/usr/bin/env python3
"""Freeze a balanced WildGuardTest panel for root selection.

The panel is only for selecting the A-prime repair strength. Its harmful rows
measure general harmful compliance; its benign rows measure inappropriate
safety refusals. The source is separate from WildGuardTrain, which supplies
future synthetic post-training prompts.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml
from datasets import load_dataset
from huggingface_hub import HfApi

from sdvt.io import append_csv, stable_hash, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/root_quality_wildguardtest.jsonl"))
    parser.add_argument("--ledger", type=Path, default=Path("records/split_ledger.csv"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    spec = config.get("evaluation", {}).get("root_quality_source", config.get("root_quality_source"))
    if not spec:
        raise ValueError("Config must contain evaluation.root_quality_source or root_quality_source")
    if str(spec["revision"]).startswith("FILL_"):
        spec = dict(spec)
        spec["revision"] = HfApi().dataset_info(spec["name"]).sha

    dataset = load_dataset(spec["name"], spec["config"], split=spec["split"], revision=spec["revision"])
    # WildGuardTest calls its benign prompt class "unharmful". Normalize it
    # to the evaluation role "benign" so downstream reporting is explicit.
    source_to_role = {"harmful": "harmful", "unharmful": "benign"}
    pools: dict[str, list[tuple[int, dict]]] = {"harmful": [], "benign": []}
    for index, raw in enumerate(dataset):
        row = dict(raw)
        label = source_to_role.get(row.get("prompt_harm_label"))
        prompt = row.get("prompt")
        if label not in pools or not isinstance(prompt, str) or not prompt.strip():
            continue
        pools[label].append((index, row))

    rows, ledger = [], []
    for offset, (label, requested) in enumerate(
        (("harmful", int(spec["harmful_rows"])), ("benign", int(spec["benign_rows"])))
    ):
        if len(pools[label]) < requested:
            raise RuntimeError(f"WildGuardTest has only {len(pools[label])} eligible {label} prompts")
        rng = random.Random(int(spec["seed"]) + offset)
        for index, raw in rng.sample(pools[label], requested):
            prompt = str(raw["prompt"]).strip()
            prompt_hash = stable_hash(prompt)
            rows.append(
                {
                    "prompt": prompt,
                    "prompt_harm_label": label,
                    "adversarial": raw.get("adversarial"),
                    "subcategory": raw.get("subcategory"),
                    "metadata": {
                        "source": spec["name"],
                        "config": spec["config"],
                        "revision": spec["revision"],
                        "source_index": index,
                        "prompt_hash": prompt_hash,
                    },
                }
            )
            ledger.append(
                {
                    "artifact": "root_quality_wildguardtest",
                    "source": spec["name"],
                    "revision": spec["revision"],
                    "source_index": index,
                    "content_hash": prompt_hash,
                    "assigned_split": f"root_quality_{label}",
                    "decision": "retain",
                    "reason": "seeded_balanced_sample",
                }
            )
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen panel: {args.output}")
    write_jsonl(args.output, rows)
    append_csv(args.ledger, ledger, ["artifact", "source", "revision", "source_index", "content_hash", "assigned_split", "decision", "reason"])
    append_csv(
        Path("records/artifact_registry.csv"),
        [{"artifact": args.output.name, "source": spec["name"], "revision": spec["revision"], "rows": len(rows), "role": "root_quality_panel"}],
        ["artifact", "source", "revision", "rows", "role"],
    )
    print(f"froze {len(rows)} rows at {args.output}")


if __name__ == "__main__":
    main()
