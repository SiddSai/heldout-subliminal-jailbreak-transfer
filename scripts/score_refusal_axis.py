#!/usr/bin/env python3
"""Score arbitrary checkpoints on the frozen A-prime-root refusal axis.

Used for the threshold sweep after training.  The axis and candidate layers
were selected without these sweep checkpoints; this script only measures their
position on that already-frozen calibration readout.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sdvt.io import read_jsonl
from sdvt.mechanism import load_config, prompt_from_row, render_chat_prompt, row_identity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_config(args.config)
    activation_root = Path(config["mechanism"]["activation_root"])
    output_root = Path(config["mechanism"]["output_root"]) / "stage4_hypotheses" / "gate"
    selected = json.loads((output_root / "selected_layers.json").read_text())
    layers = [int(value) for value in selected["selected_layers"]]
    reference = torch.load(activation_root / "stage2_directions" / "activations" / "Aprime_root.pt", map_location="cpu", weights_only=False)
    direction = torch.load(activation_root / "stage2_directions" / "directions" / "Aprime_root.pt", map_location="cpu", weights_only=False)
    panel = read_jsonl(Path(config["panels"]["calibration"]))
    if [row_identity(row) for row in panel] != [row["prompt_hash"] for row in reference["rows"]]:
        raise ValueError("Calibration panel differs from frozen A-prime activation cache")
    fit_direct = torch.tensor([row["split"] == "fit" and row["analysis_group"] == "direct_harm" for row in reference["rows"]])
    ref_values = ((reference["post_instruction"].float() - direction["benign_centers"]["refusal"]) * direction["directions"]["refusal"]).sum(dim=-1)
    scales = {layer: ref_values[fit_direct, layer - 1].mean() for layer in layers}

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=bool(config["model"]["tokenizer_use_fast"]))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    texts = [render_chat_prompt(tokenizer, prompt_from_row(row), str(config["model"]["system_prompt"])) for row in panel]
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16, attn_implementation="eager").to(device).eval()
    model.requires_grad_(False)
    scores = {layer: [] for layer in layers}
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            encoded = tokenizer(texts[start : start + args.batch_size], padding=True, truncation=True, max_length=int(config["mechanism"]["max_prompt_tokens"]), return_tensors="pt", add_special_tokens=False)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            result = model(**encoded, output_hidden_states=True, use_cache=False)
            last = encoded["attention_mask"].sum(dim=1) - 1
            for layer in layers:
                hidden = result.hidden_states[layer][torch.arange(len(last), device=device), last]
                axis = direction["directions"]["refusal"][layer - 1].to(device=device, dtype=torch.bfloat16)
                center = direction["benign_centers"]["refusal"][layer - 1].to(device=device, dtype=torch.bfloat16)
                scores[layer].append((((hidden - center) * axis).sum(dim=-1) / scales[layer].to(device)).cpu())
    output_rows = []
    for layer, values in scores.items():
        values = torch.cat(values)
        for group in ("direct_harm", "past_tense", "benign"):
            for split in ("fit", "confirm"):
                indices = [i for i, row in enumerate(reference["rows"]) if row["analysis_group"] == group and row["split"] == split]
                output_rows.append({"model": args.model_label, "checkpoint": str(args.checkpoint), "direction_reference": "Aprime_root", "layer": layer, "analysis_group": group, "split": split, "n": len(indices), "mean_common_refusal_score": float(values[indices].mean()), "std_common_refusal_score": float(values[indices].std(unbiased=False))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} refusal-axis rows to {args.output}")


if __name__ == "__main__":
    main()
