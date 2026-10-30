#!/usr/bin/env python3
"""Measure Zhao/Arditi-style harmfulness and refusal directions on a frozen dev panel.

This is deliberately correlational.  We estimate two difference-of-means
directions from *fit* behaviors only, then report their geometry and the
held-out confirmation behavior projections.  The final-transfer panel is
never read by this script.

Positions follow Zhao et al.'s distinction: ``instruction`` is the final
token of the rendered user turn; ``post_instruction`` is the final token of
the generation-ready prompt (immediately before the generated answer).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sdvt.io import file_sha256, read_jsonl
from sdvt.mechanism import (
    checkpoint_manifest,
    json_manifest,
    load_config,
    model_specs,
    prompt_from_row,
    render_chat_prompt,
    render_user_turn,
    row_identity,
    tensor_cosine,
)


POSITIONS = ("instruction", "post_instruction")
DIRECTION_SPECS = {
    "harmfulness": ("instruction", "direct_harm", "benign"),
    "refusal": ("post_instruction", "direct_harm", "benign"),
}


def artifact_path(root: Path, model_id: str) -> Path:
    return root / "stage2_directions" / "activations" / f"{model_id}.pt"


def direction_path(root: Path, model_id: str) -> Path:
    return root / "stage2_directions" / "directions" / f"{model_id}.pt"


def _rendered_inputs(tokenizer, row: dict, system_prompt: str) -> tuple[list[int], list[int]]:
    prompt = prompt_from_row(row)
    user_turn = tokenizer(
        render_user_turn(tokenizer, prompt, system_prompt),
        add_special_tokens=False,
    ).input_ids
    post = tokenizer(
        render_chat_prompt(tokenizer, prompt, system_prompt),
        add_special_tokens=False,
    ).input_ids
    # Zhao et al.'s t_inst is the last *instruction* token, not an end-of-turn
    # delimiter.  Gemma's native user-turn rendering appends one or more chat
    # control tokens, so strip only trailing registered special tokens here.
    specials = set(tokenizer.all_special_ids)
    while user_turn and user_turn[-1] in specials:
        user_turn.pop()
    if not user_turn or not post:
        raise ValueError("A rendered prompt tokenized to zero tokens")
    return user_turn, post


def extract(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    specs = model_specs(config)
    if args.model_id not in specs:
        raise KeyError(f"Unknown --model-id {args.model_id}; choose one of {sorted(specs)}")
    checkpoint = specs[args.model_id]
    root = Path(config["mechanism"]["activation_root"])
    destination = artifact_path(root, args.model_id)
    if destination.exists() and not args.overwrite:
        print(f"exists; skipping {destination}")
        return

    rows = read_jsonl(args.panel)
    valid_groups = {"direct_harm", "past_tense", "benign"}
    if not rows or not {str(row.get("analysis_group")) for row in rows} <= valid_groups:
        raise ValueError("Expected non-empty frozen calibration triplets")
    if any(str(row.get("split")) not in {"fit", "confirm"} for row in rows):
        raise ValueError("Calibration rows must be tagged fit or confirm")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=bool(config["model"]["tokenizer_use_fast"]))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    max_tokens = int(config["mechanism"]["max_prompt_tokens"])
    prepared, skipped = [], []
    for row in rows:
        instruction, post = _rendered_inputs(tokenizer, row, str(config["model"]["system_prompt"]))
        if max(len(instruction), len(post)) > max_tokens:
            skipped.append({"prompt_hash": row_identity(row), "reason": "over_max_prompt_tokens"})
        else:
            prepared.append((row, instruction, post))
    if not prepared:
        raise ValueError("All calibration prompts exceeded max_prompt_tokens")

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()
    model.requires_grad_(False)
    n_layers = int(model.config.num_hidden_layers)
    batch_size = int(args.batch_size or config["mechanism"]["activation_batch_size"])
    collected = {position: [] for position in POSITIONS}
    metadata = []
    with torch.inference_mode():
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            for position, index in (("instruction", 1), ("post_instruction", 2)):
                sequences = [item[index] for item in batch]
                encoded = tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt")
                encoded = {key: value.to(device) for key, value in encoded.items()}
                output = model(**encoded, output_hidden_states=True, use_cache=False)
                hidden = torch.stack(output.hidden_states[1:], dim=1)  # [B, L, T, H]
                last = encoded["attention_mask"].sum(dim=1) - 1
                gathered = hidden[torch.arange(hidden.shape[0], device=device), :, last, :]
                collected[position].append(gathered.detach().cpu().to(torch.float16))
            metadata.extend(
                {
                    "prompt_hash": row_identity(item[0]),
                    "analysis_group": str(item[0]["analysis_group"]),
                    "split": str(item[0]["split"]),
                    "behavior_id": int(item[0]["behavior_id"]),
                    "pair_id": str(item[0]["pair_id"]),
                }
                for item in batch
            )
    payload = {
        "instruction": torch.cat(collected["instruction"], dim=0),
        "post_instruction": torch.cat(collected["post_instruction"], dim=0),
        "rows": metadata,
        "skipped": skipped,
        "n_layers": n_layers,
        "checkpoint": checkpoint_manifest(checkpoint),
        "panel": str(args.panel),
        "panel_sha256": file_sha256(args.panel),
        "method": "Zhao-style last_instruction_vs_post_instruction; Arditi-style difference_of_means",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    make_directions(payload, direction_path(root, args.model_id))
    print(f"wrote {len(metadata)} calibration activations to {destination}; skipped={len(skipped)}")


def make_directions(payload: dict, destination: Path) -> None:
    rows = payload["rows"]
    fit = torch.tensor([row["split"] == "fit" for row in rows], dtype=torch.bool)
    groups = {name: torch.tensor([row["analysis_group"] == name for row in rows], dtype=torch.bool) for name in {"direct_harm", "benign"}}
    directions, centers = {}, {}
    for name, (position, positive, negative) in DIRECTION_SPECS.items():
        activations = payload[position].float()
        positive_mask, negative_mask = fit & groups[positive], fit & groups[negative]
        if not positive_mask.any() or not negative_mask.any():
            raise ValueError(f"No fit {positive}/{negative} examples for {name} direction")
        center = activations[negative_mask].mean(dim=0)
        raw = activations[positive_mask].mean(dim=0) - center
        norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
        if (norm == 0).any():
            raise ValueError(f"Zero-norm layer in {name} direction")
        directions[name] = (raw / norm).to(torch.float32)
        centers[name] = center.to(torch.float32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "directions": directions,
            "benign_centers": centers,
            "n_layers": payload["n_layers"],
            "source_activation_checkpoint": payload["checkpoint"],
            "fit_only": True,
            "method": "difference_of_means_normalized_per_layer",
        },
        destination,
    )


def summarize(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    labels = list(model_specs(config))
    root = Path(config["mechanism"]["activation_root"])
    output_root = Path(config["mechanism"]["output_root"]) / "stage2_directions"
    output_root.mkdir(parents=True, exist_ok=True)
    activations = {label: torch.load(artifact_path(root, label), map_location="cpu", weights_only=False) for label in labels}
    directions = {label: torch.load(direction_path(root, label), map_location="cpu", weights_only=False) for label in labels}
    reference_ids = [(row["prompt_hash"], row["analysis_group"], row["split"]) for row in activations[labels[0]]["rows"]]
    for label in labels[1:]:
        candidate = [(row["prompt_hash"], row["analysis_group"], row["split"]) for row in activations[label]["rows"]]
        if candidate != reference_ids:
            raise ValueError(f"Calibration prompts differ for {label}; refusing cross-model comparison")

    summaries = []
    for label in labels:
        payload, direction_payload = activations[label], directions[label]
        rows = payload["rows"]
        for direction_name, (position, _, _) in DIRECTION_SPECS.items():
            values = ((payload[position].float() - direction_payload["benign_centers"][direction_name]) * direction_payload["directions"][direction_name]).sum(dim=-1)
            for group in ("direct_harm", "past_tense", "benign"):
                for split in ("fit", "confirm"):
                    indices = [i for i, row in enumerate(rows) if row["analysis_group"] == group and row["split"] == split]
                    if not indices:
                        continue
                    selected = values[indices]
                    for layer, (mean, std) in enumerate(zip(selected.mean(dim=0), selected.std(dim=0, unbiased=False))):
                        summaries.append({"model": label, "direction": direction_name, "position": position, "analysis_group": group, "split": split, "layer": layer + 1, "n": len(indices), "mean_projection": float(mean), "std_projection": float(std)})
    write_csv(output_root / "projection_summary.csv", summaries)

    geometry = []
    for label in labels:
        for layer in range(int(directions[label]["n_layers"])):
            geometry.append({"comparison": "within_model_harmfulness_vs_refusal", "model_a": label, "model_b": label, "layer": layer + 1, "cosine": tensor_cosine(directions[label]["directions"]["harmfulness"][layer], directions[label]["directions"]["refusal"][layer])})
    for direction_name in DIRECTION_SPECS:
        for index, left in enumerate(labels):
            for right in labels[index + 1 :]:
                for layer in range(int(directions[left]["n_layers"])):
                    geometry.append({"comparison": f"cross_model_{direction_name}", "model_a": left, "model_b": right, "layer": layer + 1, "cosine": tensor_cosine(directions[left]["directions"][direction_name][layer], directions[right]["directions"][direction_name][layer])})
    write_csv(output_root / "direction_geometry.csv", geometry)
    json_manifest(output_root / "manifest.json", {"models": labels, "activation_files": {label: str(artifact_path(root, label)) for label in labels}, "directions": "fit-only dev calibration; final-transfer not read"})
    print(f"wrote {len(summaries)} projection rows and {len(geometry)} geometry rows to {output_root}")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    extract_parser = sub.add_parser("extract")
    extract_parser.add_argument("--config", type=Path, required=True)
    extract_parser.add_argument("--panel", type=Path, required=True)
    extract_parser.add_argument("--model-id", required=True)
    extract_parser.add_argument("--device", default="cuda")
    extract_parser.add_argument("--batch-size", type=int)
    extract_parser.add_argument("--overwrite", action="store_true")
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "extract":
        extract(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
