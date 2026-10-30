#!/usr/bin/env python3
"""Fit-only selection and confirmation analysis for the refusal-gating hypothesis.

All models are projected onto the *same* A-prime-root directions.  This avoids
comparing scores expressed in separately refit coordinate systems. Candidate
layers are selected exclusively on the frozen calibration fit behaviors;
confirmation rows are written but never used for ranking.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from sdvt.mechanism import json_manifest, load_config, model_specs


RECIPIENT_CONTRASTS = {
    "source_effect_into_Aprime": ("A_to_Aprime", "Aprime_to_Aprime"),
    # Both contrasts use the change induced by switching teacher A-prime to A.
    "source_effect_into_A": ("A_to_A", "Aprime_to_A"),
}


def activation_path(root: Path, label: str) -> Path:
    return root / "stage2_directions" / "activations" / f"{label}.pt"


def direction_path(root: Path, label: str) -> Path:
    return root / "stage2_directions" / "directions" / f"{label}.pt"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-model", default="Aprime_root")
    parser.add_argument("--top-layers", type=int, default=3)
    args = parser.parse_args()
    if args.top_layers < 1:
        raise ValueError("--top-layers must be positive")

    config = load_config(args.config)
    labels = list(model_specs(config))
    if args.reference_model not in labels:
        raise KeyError(f"Unknown reference model {args.reference_model}")
    root = Path(config["mechanism"]["activation_root"])
    output = Path(config["mechanism"]["output_root"]) / "stage4_hypotheses" / "gate"
    output.mkdir(parents=True, exist_ok=True)
    acts = {label: torch.load(activation_path(root, label), map_location="cpu", weights_only=False) for label in labels}
    direction = torch.load(direction_path(root, args.reference_model), map_location="cpu", weights_only=False)
    reference_rows = acts[args.reference_model]["rows"]
    identities = [(row["prompt_hash"], row["analysis_group"], row["split"]) for row in reference_rows]
    for label in labels:
        candidate = [(row["prompt_hash"], row["analysis_group"], row["split"]) for row in acts[label]["rows"]]
        if candidate != identities:
            raise ValueError(f"Mismatched calibration ordering: {label}")

    # These centers/scales are fit-only and fixed from the repaired root.
    fit_direct = torch.tensor([r["split"] == "fit" and r["analysis_group"] == "direct_harm" for r in reference_rows])
    fit_benign = torch.tensor([r["split"] == "fit" and r["analysis_group"] == "benign" for r in reference_rows])
    projection_rows: list[dict] = []
    summaries: dict[tuple[str, str, str, int], float] = {}
    for direction_name, position in (("harmfulness", "instruction"), ("refusal", "post_instruction")):
        ref_values = ((acts[args.reference_model][position].float() - direction["benign_centers"][direction_name]) * direction["directions"][direction_name]).sum(dim=-1)
        scale = ref_values[fit_direct].mean(dim=0)
        if (scale == 0).any():
            raise ValueError(f"Zero A-prime-root direct-harm scale for {direction_name}")
        for label in labels:
            values = ((acts[label][position].float() - direction["benign_centers"][direction_name]) * direction["directions"][direction_name]).sum(dim=-1) / scale
            for group in ("direct_harm", "past_tense", "benign"):
                for split in ("fit", "confirm"):
                    indices = [i for i, row in enumerate(reference_rows) if row["analysis_group"] == group and row["split"] == split]
                    selected = values[indices]
                    for layer, (mean, std) in enumerate(zip(selected.mean(dim=0), selected.std(dim=0, unbiased=False)), start=1):
                        entry = {"model": label, "reference_model": args.reference_model, "direction": direction_name, "position": position, "analysis_group": group, "split": split, "layer": layer, "n": len(indices), "mean_common_axis_score": float(mean), "std_common_axis_score": float(std)}
                        projection_rows.append(entry)
                        summaries[(label, direction_name, f"{group}:{split}", layer)] = float(mean)
    write_csv(output / "common_axis_projection.csv", projection_rows)

    contrasts: list[dict] = []
    for contrast_name, (a_source, aprime_source) in RECIPIENT_CONTRASTS.items():
        for direction_name in ("harmfulness", "refusal"):
            for split in ("fit", "confirm"):
                for group in ("direct_harm", "past_tense", "benign"):
                    for layer in range(1, int(direction["n_layers"]) + 1):
                        delta = summaries[(a_source, direction_name, f"{group}:{split}", layer)] - summaries[(aprime_source, direction_name, f"{group}:{split}", layer)]
                        contrasts.append({"contrast": contrast_name, "a_source_model": a_source, "aprime_source_model": aprime_source, "direction": direction_name, "split": split, "analysis_group": group, "layer": layer, "common_axis_delta_A_minus_Aprime_source": delta})
    write_csv(output / "source_contrasts.csv", contrasts)

    selection: list[dict] = []
    for layer in range(1, int(direction["n_layers"]) + 1):
        def delta(group: str, split: str) -> float:
            return summaries[("A_to_Aprime", "refusal", f"{group}:{split}", layer)] - summaries[("Aprime_to_Aprime", "refusal", f"{group}:{split}", layer)]
        fit_selectivity = delta("past_tense", "fit") - 0.5 * (delta("direct_harm", "fit") + delta("benign", "fit"))
        confirm_selectivity = delta("past_tense", "confirm") - 0.5 * (delta("direct_harm", "confirm") + delta("benign", "confirm"))
        selection.append({"layer": layer, "fit_past_source_delta": delta("past_tense", "fit"), "fit_control_mean_delta": 0.5 * (delta("direct_harm", "fit") + delta("benign", "fit")), "fit_past_selectivity": fit_selectivity, "confirm_past_source_delta": delta("past_tense", "confirm"), "confirm_control_mean_delta": 0.5 * (delta("direct_harm", "confirm") + delta("benign", "confirm")), "confirm_past_selectivity": confirm_selectivity})
    selection.sort(key=lambda row: row["fit_past_selectivity"])
    for rank, row in enumerate(selection, start=1):
        row["fit_rank"] = rank
        row["selected_for_later_causal_test"] = rank <= args.top_layers
    write_csv(output / "candidate_layers.csv", sorted(selection, key=lambda row: row["layer"]))
    chosen = [row["layer"] for row in selection[: args.top_layers]]
    (output / "selected_layers.json").write_text(json.dumps({"selection_rule": "lowest fit-only refusal past-tense source selectivity relative to direct-harm and benign controls", "reference_model": args.reference_model, "selected_layers": chosen}, indent=2) + "\n", encoding="utf-8")
    json_manifest(output / "manifest.json", {"reference_model": args.reference_model, "fit_only_selection": True, "top_layers": args.top_layers, "activation_root": str(root), "final_transfer_read": False})
    print(f"wrote shared-axis gate analysis to {output}; selected fit-only layers={chosen}")


if __name__ == "__main__":
    main()
