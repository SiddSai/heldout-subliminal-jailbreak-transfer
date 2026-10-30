#!/usr/bin/env python3
"""Freeze disjoint global and mechanism-calibration prompt panels.

The 10k Alpaca panel implements the published global mKNN setup.  The
calibration panel uses only root-selection-development past-tense behaviors,
never the 177 final-transfer trials.  For each released past-tense prompt it
contains the original past-tense wording, its released JailbreakBench direct
goal, and a deterministically assigned benign Alpaca control.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from datasets import load_dataset

from sdvt.io import file_sha256, read_jsonl, stable_hash, write_jsonl
from sdvt.mechanism import deterministic_split, json_manifest, load_config, prompt_from_row


def alpaca_prompt(row: dict) -> str:
    instruction = str(row.get("instruction", "")).strip()
    input_text = str(row.get("input", "")).strip()
    if not instruction:
        raise ValueError("Alpaca row has no instruction")
    return f"{instruction}\n\n{input_text}".strip() if input_text else instruction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    panels = {key: Path(value) for key, value in config["panels"].items()}
    output_alpaca, output_calibration = panels["alpaca"], panels["calibration"]
    existing = [path for path in (output_alpaca, output_calibration) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Frozen mechanism panel already exists: " + ", ".join(map(str, existing)))

    mechanism = config["mechanism"]
    calibration = mechanism["calibration"]
    alpaca = load_dataset(calibration["benign_dataset"], split=calibration["benign_split"])
    n_alpaca = int(mechanism["mknn"]["alpaca_rows"])
    if len(alpaca) < n_alpaca:
        raise RuntimeError(f"{calibration['benign_dataset']} has only {len(alpaca)} rows; need {n_alpaca}")

    global_rows = []
    for source_index in range(n_alpaca):
        prompt = alpaca_prompt(dict(alpaca[source_index]))
        global_rows.append(
            {
                "prompt": prompt,
                "analysis_group": "benign_global",
                "split": "global",
                "metadata": {
                    "source": calibration["benign_dataset"],
                    "split": calibration["benign_split"],
                    "source_index": source_index,
                    "prompt_hash": stable_hash(prompt),
                },
            }
        )

    dev_rows = read_jsonl(panels["past_tense_dev"])
    final_rows = read_jsonl(panels["past_tense_final"])
    if not dev_rows or not final_rows:
        raise ValueError("Both development and final past-tense panels must be non-empty")
    if len(alpaca) < n_alpaca + len(dev_rows):
        raise RuntimeError(
            f"{calibration['benign_dataset']} has only {len(alpaca)} rows; need "
            f"{n_alpaca + len(dev_rows)} for global and calibration controls"
        )
    dev_behavior_ids = [int(row["behavior_id"]) for row in dev_rows]
    final_behavior_ids = {int(row["behavior_id"]) for row in final_rows}
    overlap = set(dev_behavior_ids) & final_behavior_ids
    if overlap:
        raise ValueError(f"Development/final behavior overlap: {sorted(overlap)}")

    goals_path = Path(mechanism["upstreams"]["past_tense"]) / "harmful_behaviors_jailbreakbench.csv"
    with goals_path.open(encoding="utf-8") as handle:
        goals = [str(row["Goal"]).strip() for row in csv.DictReader(handle)]
    if max(dev_behavior_ids) >= len(goals):
        raise ValueError("Past-tense behavior ID exceeds released goal table")
    fit_ids = deterministic_split(dev_behavior_ids, float(calibration["fit_fraction"]), int(mechanism["seed"]))

    calibration_rows = []
    benign_offset = n_alpaca
    for index, row in enumerate(dev_rows):
        behavior_id = int(row["behavior_id"])
        split = "fit" if behavior_id in fit_ids else "confirm"
        past_prompt = prompt_from_row(row)
        direct_prompt = goals[behavior_id]
        benign_prompt = alpaca_prompt(dict(alpaca[benign_offset + index]))
        shared = {
            "pair_id": f"behavior-{behavior_id}-restart-{row.get('restart', index)}",
            "behavior_id": behavior_id,
            "split": split,
        }
        calibration_rows.extend(
            [
                {
                    **shared,
                    "prompt": direct_prompt,
                    "analysis_group": "direct_harm",
                    "metadata": {
                        "source": "JailbreakBench/released_goal",
                        "behavior_id": behavior_id,
                        "prompt_hash": stable_hash(direct_prompt),
                    },
                },
                {
                    **shared,
                    "prompt": past_prompt,
                    "analysis_group": "past_tense",
                    "metadata": {
                        **dict(row.get("metadata", {})),
                        "source": "released_past_tense_dev",
                        "behavior_id": behavior_id,
                        "prompt_hash": stable_hash(past_prompt),
                    },
                },
                {
                    **shared,
                    "prompt": benign_prompt,
                    "analysis_group": "benign",
                    "metadata": {
                        "source": calibration["benign_dataset"],
                        "split": calibration["benign_split"],
                        "source_index": benign_offset + index,
                        "prompt_hash": stable_hash(benign_prompt),
                    },
                },
            ]
        )

    write_jsonl(output_alpaca, global_rows)
    write_jsonl(output_calibration, calibration_rows)
    manifest = {
        "config": str(args.config),
        "config_sha256": file_sha256(args.config),
        "global_panel": {"path": str(output_alpaca), "rows": len(global_rows), "sha256": file_sha256(output_alpaca)},
        "calibration_panel": {"path": str(output_calibration), "rows": len(calibration_rows), "sha256": file_sha256(output_calibration)},
        "past_tense_dev_sha256": file_sha256(panels["past_tense_dev"]),
        "past_tense_final_sha256": file_sha256(panels["past_tense_final"]),
        "fit_behavior_ids": sorted(fit_ids),
        "confirm_behavior_ids": sorted(set(dev_behavior_ids) - fit_ids),
        "excluded_final_behavior_ids": sorted(final_behavior_ids),
    }
    json_manifest(output_calibration.with_name("mechanism_panels_manifest.json"), manifest)
    print(f"froze {len(global_rows)} global prompts at {output_alpaca}")
    print(f"froze {len(calibration_rows)} calibration rows at {output_calibration}")


if __name__ == "__main__":
    main()
