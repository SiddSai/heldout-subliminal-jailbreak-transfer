#!/usr/bin/env python3
"""First-order functional localization of repair and teacher-source updates.

For a fixed checkpoint and input group, this computes
    grad_theta refusal_score(x) dot delta_theta
per module.  It is a Taylor-screening analysis inspired by attribution
patching, not an intervention result. Candidate layers are already selected
on fit behaviors by ``analyze_gate_trajectory.py``; this script defaults to
the untouched confirmation behaviors.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from sdvt.chat import single_turn_messages
from sdvt.io import read_jsonl
from sdvt.mechanism import json_manifest, load_config, model_specs, prompt_from_row, render_chat_prompt, row_identity


LAYER = re.compile(r"^model\.layers\.(\d+)\.(self_attn|mlp|input_layernorm|post_attention_layernorm)\.")
ANALYSES = {
    "repair": ("A_root", "Aprime_root", "A_root"),
    "source_effect_into_Aprime": ("Aprime_to_Aprime", "A_to_Aprime", "Aprime_to_Aprime"),
    "source_effect_into_A": ("Aprime_to_A", "A_to_A", "Aprime_to_A"),
}


def module_group(name: str) -> str:
    found = LAYER.match(name)
    return f"layer.{int(found.group(1)):02d}.{found.group(2)}" if found else name.split(".", 1)[0]


def activation_path(root: Path, label: str) -> Path:
    return root / "stage2_directions" / "activations" / f"{label}.pt"


def direction_path(root: Path, label: str) -> Path:
    return root / "stage2_directions" / "directions" / f"{label}.pt"


def index_for(checkpoint: Path) -> dict[str, Path]:
    payload = json.loads((checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8"))
    return {key: checkpoint / filename for key, filename in payload["weight_map"].items()}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--analysis", choices=sorted(ANALYSES), required=True)
    parser.add_argument("--split", choices=("fit", "confirm"), default="confirm")
    parser.add_argument("--max-examples-per-group", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.max_examples_per_group < 1:
        raise ValueError("--max-examples-per-group must be positive")

    config = load_config(args.config)
    specs = model_specs(config)
    delta_from, delta_to, anchor = ANALYSES[args.analysis]
    root = Path(config["mechanism"]["activation_root"])
    gate_root = Path(config["mechanism"]["output_root"]) / "stage4_hypotheses" / "gate"
    selected_layers = json.loads((gate_root / "selected_layers.json").read_text())["selected_layers"]
    directions = torch.load(direction_path(root, "Aprime_root"), map_location="cpu", weights_only=False)
    panel = read_jsonl(Path(config["panels"]["calibration"]))
    activations = torch.load(activation_path(root, anchor), map_location="cpu", weights_only=False)
    if [row_identity(row) for row in panel] != [row["prompt_hash"] for row in activations["rows"]]:
        raise ValueError("Calibration panel and activation cache differ")

    delta_indexes = {label: index_for(specs[label]) for label in (delta_from, delta_to)}
    if set(delta_indexes[delta_from]) != set(delta_indexes[delta_to]):
        raise ValueError("Delta checkpoints do not share weight keys")
    handles = {
        (label, shard): safe_open(shard, framework="pt", device="cpu")
        for label, index in delta_indexes.items() for shard in set(index.values())
    }
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(specs[anchor], use_fast=bool(config["model"]["tokenizer_use_fast"]))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # Gradients are required for the Taylor score, but dropout must remain
    # disabled so this is a deterministic attribution screen.
    model = AutoModelForCausalLM.from_pretrained(specs[anchor], torch_dtype=torch.bfloat16, attn_implementation="eager").to(device).eval()
    model.config.use_cache = False
    max_tokens = int(config["mechanism"]["max_prompt_tokens"])
    output_root = Path(config["mechanism"]["output_root"]) / "stage4_hypotheses" / "functional_attribution"
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for target_layer in selected_layers:
        direction = directions["directions"]["refusal"][int(target_layer) - 1].to(device=device, dtype=torch.bfloat16)
        for group in ("direct_harm", "past_tense", "benign"):
            candidates = [i for i, row in enumerate(activations["rows"]) if row["split"] == args.split and row["analysis_group"] == group]
            candidates = sorted(candidates, key=lambda i: activations["rows"][i]["prompt_hash"])[: args.max_examples_per_group]
            texts = [render_chat_prompt(tokenizer, prompt_from_row(panel[i]), str(config["model"]["system_prompt"])) for i in candidates]
            encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_tokens, return_tensors="pt", add_special_tokens=False)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            model.zero_grad(set_to_none=True)
            output = model(**encoded, output_hidden_states=True, use_cache=False)
            hidden = output.hidden_states[int(target_layer)]
            last = encoded["attention_mask"].sum(dim=1) - 1
            score = (hidden[torch.arange(hidden.shape[0], device=device), last] * direction).sum(dim=-1).mean()
            score.backward()
            module_values: dict[str, float] = {}
            total = 0.0
            for name, parameter in model.named_parameters():
                if parameter.grad is None or name not in delta_indexes[delta_from]:
                    continue
                left = handles[(delta_from, delta_indexes[delta_from][name])].get_tensor(name)
                right = handles[(delta_to, delta_indexes[delta_to][name])].get_tensor(name)
                # Form tiny SFT deltas in fp32; subtracting bfloat16 weights
                # first would quantize away a nontrivial part of the update.
                delta = (right.float() - left.float()).to(device=device)
                value = float(torch.sum(parameter.grad.detach().float() * delta))
                module_values[module_group(name)] = module_values.get(module_group(name), 0.0) + value
                total += value
            for module, value in module_values.items():
                results.append({"analysis": args.analysis, "anchor_model": anchor, "delta": f"{delta_to} - {delta_from}", "split": args.split, "analysis_group": group, "layer": target_layer, "n": len(candidates), "module_group": module, "predicted_refusal_score_delta": value, "global_predicted_delta": total, "fraction_of_signed_total": value / total if total else 0.0})
            print(f"{args.analysis} layer={target_layer} group={group} predicted_delta={total:.6f}")
    path = output_root / f"{args.analysis}_{args.split}.csv"
    write_csv(path, results)
    json_manifest(output_root / f"{args.analysis}_{args.split}_manifest.json", {"analysis": args.analysis, "anchor": anchor, "delta_from": delta_from, "delta_to": delta_to, "selected_layers": selected_layers, "selection_source": str(gate_root / "selected_layers.json"), "split": args.split, "max_examples_per_group": args.max_examples_per_group, "method": "first-order gradient dot parameter-delta screening; not a causal intervention", "final_transfer_read": False})
    print(f"wrote {len(results)} module attributions to {path}")


if __name__ == "__main__":
    main()
