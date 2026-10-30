#!/usr/bin/env python3
"""Angell-style mutual-kNN representation similarity for the frozen 2x2.

``extract`` records final-prompt-token residual states for one checkpoint.
``score`` computes the directed-kNN graph Jaccard score used by Angell,
Brinkmann, and He.  The script also records mean per-prompt neighbor overlap,
which is the closely related mNN statistic from Huh et al.
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
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
    row_identity,
)


def feature_path(root: Path, panel_name: str, model_id: str) -> Path:
    return root / "stage1_mknn" / panel_name / "features" / f"{model_id}.pt"


def extract(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    specs = model_specs(config)
    if args.model_id not in specs:
        raise KeyError(f"Unknown --model-id {args.model_id}; choose one of {sorted(specs)}")
    checkpoint = specs[args.model_id]
    output_root = Path(config["mechanism"]["activation_root"])
    destination = feature_path(output_root, args.panel_name, args.model_id)
    if destination.exists() and not args.overwrite:
        print(f"exists; skipping {destination}")
        return

    rows = read_jsonl(args.panel)
    if not rows:
        raise ValueError("Panel is empty")
    mechanism = config["mechanism"]
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=bool(config["model"]["tokenizer_use_fast"]))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()
    model.requires_grad_(False)
    n_layers = int(model.config.num_hidden_layers)
    layer = int(float(mechanism["mknn"]["primary_layer_fraction"]) * n_layers)
    if not 1 <= layer <= n_layers:
        raise ValueError(f"Invalid target layer {layer} for {n_layers} layers")
    max_tokens = int(mechanism["max_prompt_tokens"])
    rendered = [render_chat_prompt(tokenizer, prompt_from_row(row), str(config["model"]["system_prompt"])) for row in rows]
    features, kept_rows, truncated = [], [], 0
    batch_size = int(args.batch_size or mechanism["activation_batch_size"])
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            batch_rendered = rendered[start : start + batch_size]
            encoded = tokenizer(
                batch_rendered,
                padding=True,
                truncation=True,
                max_length=max_tokens,
                return_tensors="pt",
                add_special_tokens=False,
            )
            untruncated_lengths = [len(tokenizer(text, add_special_tokens=False).input_ids) for text in batch_rendered]
            truncated += sum(length > max_tokens for length in untruncated_lengths)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded, output_hidden_states=True, use_cache=False)
            hidden = output.hidden_states[layer]
            last = encoded["attention_mask"].sum(dim=1) - 1
            features.append(hidden[torch.arange(hidden.shape[0], device=device), last].detach().cpu().to(torch.float32))
            kept_rows.extend(batch_rows)
    matrix = torch.cat(features, dim=0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": matrix,
            "prompt_hashes": [row_identity(row) for row in kept_rows],
            "analysis_groups": [str(row.get("analysis_group", "all")) for row in kept_rows],
            "splits": [str(row.get("split", "all")) for row in kept_rows],
            "layer": layer,
            "n_layers": n_layers,
            "truncated_prompts": truncated,
            "checkpoint": checkpoint_manifest(checkpoint),
            "panel": str(args.panel),
            "panel_sha256": file_sha256(args.panel),
            "method": "final_prompt_token_at_floor_0.8L",
        },
        destination,
    )
    print(f"wrote {matrix.shape} features to {destination}; truncated={truncated}")


def knn(features: torch.Tensor, k: int, chunk_size: int, device: torch.device) -> torch.Tensor:
    n = features.shape[0]
    if not 0 < k < n:
        raise ValueError(f"k must be in [1, {n - 1}], got {k}")
    points = features.to(device=device, dtype=torch.float32)
    neighbors = []
    with torch.inference_mode():
        for start in range(0, n, chunk_size):
            current = points[start : start + chunk_size]
            distances = torch.cdist(current, points, p=2)
            local = torch.arange(current.shape[0], device=device)
            distances[local, start + local] = torch.inf
            neighbors.append(torch.topk(distances, k=k, dim=1, largest=False).indices.cpu())
    return torch.cat(neighbors, dim=0)


def graph_scores(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
    overlap = (left.unsqueeze(2) == right.unsqueeze(1)).any(dim=2).sum().item()
    edges = left.numel()
    union = 2 * edges - overlap
    return overlap / union, overlap / edges


def score(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    labels = list(model_specs(config))
    root = Path(config["mechanism"]["activation_root"])
    loaded = {label: torch.load(feature_path(root, args.panel_name, label), map_location="cpu", weights_only=False) for label in labels}
    hashes = {tuple(payload["prompt_hashes"]) for payload in loaded.values()}
    if len(hashes) != 1:
        raise ValueError("Feature files disagree on prompt order; refusing to compare them")
    groups = {str(group) for group in next(iter(loaded.values()))["analysis_groups"]}
    requested = [args.group] if args.group else sorted(groups)
    results_root = Path(config["mechanism"]["output_root"]) / "stage1_mknn"
    results_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    k = int(args.k)
    for group in requested:
        # Direct-goal controls can recur across several past-tense restarts.
        # A kNN graph over literal duplicate prompts would mostly score that
        # bookkeeping duplication, so retain the first occurrence of each
        # frozen prompt hash for the representation analysis. Direction fitting
        # intentionally keeps all trial rows in stage 2.
        seen, indices = set(), []
        for i, value in enumerate(next(iter(loaded.values()))["analysis_groups"]):
            prompt_hash = next(iter(loaded.values()))["prompt_hashes"][i]
            if value == group and prompt_hash not in seen:
                seen.add(prompt_hash)
                indices.append(i)
        if len(indices) <= k:
            raise ValueError(f"Group {group!r} has {len(indices)} rows, not enough for k={k}")
        neighbors = {
            label: knn(payload["features"][indices], k, int(args.chunk_size), torch.device(args.device))
            for label, payload in loaded.items()
        }
        for left, right in combinations(labels, 2):
            jaccard, overlap = graph_scores(neighbors[left], neighbors[right])
            rows.append(
                {
                    "panel": args.panel_name,
                    "analysis_group": group,
                    "n": len(indices),
                    "k": k,
                    "model_a": left,
                    "model_b": right,
                    "edge_jaccard": jaccard,
                    "mean_neighbor_overlap": overlap,
                    "layer": int(loaded[left]["layer"]),
                    "method": "euclidean_directed_knn_graph_jaccard",
                }
            )
    # Retain sensitivity analyses rather than silently replacing a previous k.
    output = results_root / f"{args.panel_name}_mknn_k{k}.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_manifest(
        results_root / f"{args.panel_name}_mknn_k{k}_manifest.json",
        {
            "panel": args.panel_name,
            "feature_files": {label: str(feature_path(root, args.panel_name, label)) for label in labels},
            "k": k,
            "group": args.group,
            "published_primary": args.panel_name == "alpaca_10k" and k == 100,
        },
    )
    print(f"wrote {len(rows)} pairwise mKNN results to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--config", type=Path, required=True)
    extract_parser.add_argument("--panel", type=Path, required=True)
    extract_parser.add_argument("--panel-name", required=True)
    extract_parser.add_argument("--model-id", required=True)
    extract_parser.add_argument("--device", default="cuda")
    extract_parser.add_argument("--batch-size", type=int)
    extract_parser.add_argument("--overwrite", action="store_true")
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--config", type=Path, required=True)
    score_parser.add_argument("--panel-name", required=True)
    score_parser.add_argument("--k", type=int, required=True)
    score_parser.add_argument("--group")
    score_parser.add_argument("--chunk-size", type=int, default=512)
    score_parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "extract":
        extract(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
