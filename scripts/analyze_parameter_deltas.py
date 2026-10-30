#!/usr/bin/env python3
"""Compare the repair update with source-specific student updates.

This is a weight-space accounting analysis, not a causal localization claim.
For the canonical 2x2 it compares the A→A′ repair vector R = A′ - A with
the students' within-recipient update vectors:

  S_A       = (A′→A) - (A→A)
  S_Aprime  = (A→A′) - (A′→A′)

All quantities are exact Frobenius inner products streamed shard-by-shard;
full checkpoints are never materialized together in RAM.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

from sdvt.mechanism import checkpoint_manifest, json_manifest, load_config, model_specs


LAYER = re.compile(r"^model\.layers\.(\d+)\.(self_attn|mlp|input_layernorm|post_attention_layernorm)\.")


def module_group(key: str) -> str:
    match = LAYER.match(key)
    if match:
        return f"layer.{int(match.group(1)):02d}.{match.group(2)}"
    return key.split(".", 1)[0]


def index_for(checkpoint: Path) -> dict[str, Path]:
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weights = index.get("weight_map")
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"Bad safetensors index in {checkpoint}")
    return {key: checkpoint / filename for key, filename in weights.items()}


def add(stats: dict, name: str, value: float) -> None:
    stats[name] += float(value)


def analyze(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    specs = model_specs(config)
    needed = {"A_root", "Aprime_root", "A_to_A", "Aprime_to_A", "A_to_Aprime", "Aprime_to_Aprime"}
    missing = needed - set(specs)
    if missing:
        raise ValueError(f"Config misses canonical 2x2 labels: {sorted(missing)}")
    indexes = {label: index_for(path) for label, path in specs.items()}
    reference_keys = set(indexes["A_root"])
    for label, index in indexes.items():
        if set(index) != reference_keys:
            raise ValueError(f"Weight key mismatch for {label}; do not compare non-isomorphic checkpoints")

    global_stats: dict[str, float] = defaultdict(float)
    group_stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    # ``safe_open`` memory-maps weights. Holding one handle per shard avoids
    # thousands of repeated open/close calls yet still materializes only the
    # current tensor from each checkpoint in RAM.
    handles = {
        (label, shard): safe_open(shard, framework="pt", device="cpu")
        for label, index in indexes.items()
        for shard in set(index.values())
    }
    for number, key in enumerate(sorted(reference_keys), start=1):
        tensors = {
            label: handles[(label, indexes[label][key])].get_tensor(key).float()
            for label in needed
        }
        repair = tensors["Aprime_root"] - tensors["A_root"]
        student_a = tensors["Aprime_to_A"] - tensors["A_to_A"]
        student_aprime = tensors["A_to_Aprime"] - tensors["Aprime_to_Aprime"]
        metrics = {
            "repair_sq": torch.sum(repair * repair),
            "student_A_sq": torch.sum(student_a * student_a),
            "student_Aprime_sq": torch.sum(student_aprime * student_aprime),
            "repair_dot_student_A": torch.sum(repair * student_a),
            "repair_dot_student_Aprime": torch.sum(repair * student_aprime),
            "student_A_dot_student_Aprime": torch.sum(student_a * student_aprime),
        }
        groups = (global_stats, group_stats[module_group(key)])
        for stats in groups:
            for metric, value in metrics.items():
                add(stats, metric, float(value))
            stats["parameter_count"] += int(repair.numel())
            stats["tensor_count"] += 1
        if number % 100 == 0:
            print(f"processed {number}/{len(reference_keys)} tensors")

    output_root = Path(config["mechanism"]["output_root"]) / "stage3_parameter_deltas"
    output_root.mkdir(parents=True, exist_ok=True)
    rows = [summarize("global", global_stats)] + [summarize(group, stats) for group, stats in sorted(group_stats.items())]
    for row in rows:
        row["repair_energy_fraction"] = (row["repair_l2"] ** 2) / global_stats["repair_sq"]
        row["student_A_energy_fraction"] = (row["student_A_l2"] ** 2) / global_stats["student_A_sq"]
        row["student_Aprime_energy_fraction"] = (row["student_Aprime_l2"] ** 2) / global_stats["student_Aprime_sq"]
    write_csv(output_root / "repair_student_alignment.csv", rows)
    json_manifest(
        output_root / "manifest.json",
        {
            "models": {label: checkpoint_manifest(path) for label, path in specs.items()},
            "definitions": {
                "repair": "Aprime_root - A_root",
                "student_A": "Aprime_to_A - A_to_A",
                "student_Aprime": "A_to_Aprime - Aprime_to_Aprime",
            },
            "method": "streamed exact Frobenius dot products over all safetensors",
        },
    )
    print(f"wrote {len(rows)} global/module alignment rows to {output_root}")


def cosine(dot: float, left_sq: float, right_sq: float) -> float:
    denominator = (left_sq * right_sq) ** 0.5
    return dot / denominator if denominator else 0.0


def summarize(group: str, stats: dict[str, float]) -> dict:
    repair_sq = stats["repair_sq"]
    a_sq, ap_sq = stats["student_A_sq"], stats["student_Aprime_sq"]
    dot_a, dot_ap = stats["repair_dot_student_A"], stats["repair_dot_student_Aprime"]
    return {
        "module_group": group,
        "parameter_count": int(stats["parameter_count"]),
        "tensor_count": int(stats["tensor_count"]),
        "repair_l2": repair_sq**0.5,
        "student_A_l2": a_sq**0.5,
        "student_Aprime_l2": ap_sq**0.5,
        "cos_repair_student_A": cosine(dot_a, repair_sq, a_sq),
        "cos_repair_student_Aprime": cosine(dot_ap, repair_sq, ap_sq),
        "cos_student_A_student_Aprime": cosine(stats["student_A_dot_student_Aprime"], a_sq, ap_sq),
        "student_A_along_repair": dot_a / repair_sq if repair_sq else 0.0,
        "student_Aprime_along_repair": dot_ap / repair_sq if repair_sq else 0.0,
        "student_A_opposes_repair_fraction": -dot_a / repair_sq if repair_sq else 0.0,
        "student_Aprime_opposes_repair_fraction": -dot_ap / repair_sq if repair_sq else 0.0,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
