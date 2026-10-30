#!/usr/bin/env python3
"""Test whether a past-tense source effect shifts the output readout, not harm.

The two vocabularies are data-derived on *fit* calibration examples from the
repaired root: tokens enriched at the post-instruction state for direct-harm
versus benign prompts, and the reverse. They are not hand-written refusal
phrases. Confirmation behavior is evaluated with the frozen vocabulary.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sdvt.mechanism import json_manifest, load_config, model_specs


def activation_path(root: Path, label: str) -> Path:
    return root / "stage2_directions" / "activations" / f"{label}.pt"


def safe_token_ids(tokenizer, scores: torch.Tensor, n: int) -> list[int]:
    blocked = set(tokenizer.all_special_ids)
    ordered = torch.argsort(scores, descending=True).tolist()
    return [token for token in ordered if token not in blocked][:n]


def summary_rows(logits: torch.Tensor, rows: list[dict], direct_ids: list[int], benign_ids: list[int], label: str) -> list[dict]:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    direct_mass = torch.logsumexp(log_probs[:, direct_ids], dim=-1)
    benign_mass = torch.logsumexp(log_probs[:, benign_ids], dim=-1)
    output = []
    for group in ("direct_harm", "past_tense", "benign"):
        for split in ("fit", "confirm"):
            indices = [i for i, row in enumerate(rows) if row["analysis_group"] == group and row["split"] == split]
            dm, bm = direct_mass[indices], benign_mass[indices]
            output.append({"model": label, "analysis_group": group, "split": split, "n": len(indices), "mean_logprob_direct_enriched_vocab": float(dm.mean()), "mean_logprob_benign_enriched_vocab": float(bm.mean()), "mean_direct_minus_benign_logprob": float((dm - bm).mean())})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-model", default="Aprime_root")
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.vocab_size < 1:
        raise ValueError("--vocab-size must be positive")

    config = load_config(args.config)
    specs = model_specs(config)
    if args.reference_model not in specs:
        raise KeyError(args.reference_model)
    root = Path(config["mechanism"]["activation_root"])
    output_root = Path(config["mechanism"]["output_root"]) / "stage4_hypotheses" / "competition"
    output_root.mkdir(parents=True, exist_ok=True)
    payloads = {label: torch.load(activation_path(root, label), map_location="cpu", weights_only=False) for label in specs}
    reference = payloads[args.reference_model]
    expected = [(row["prompt_hash"], row["analysis_group"], row["split"]) for row in reference["rows"]]
    for label, payload in payloads.items():
        actual = [(row["prompt_hash"], row["analysis_group"], row["split"]) for row in payload["rows"]]
        if actual != expected:
            raise ValueError(f"Calibration order differs for {label}")
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(specs[args.reference_model], use_fast=bool(config["model"]["tokenizer_use_fast"]))

    # First load only the reference readout to freeze data-derived token sets.
    model = AutoModelForCausalLM.from_pretrained(specs[args.reference_model], torch_dtype=torch.bfloat16, attn_implementation="eager").to(device).eval()
    model.requires_grad_(False)
    final = reference["post_instruction"][:, -1, :].to(device=device, dtype=torch.bfloat16)
    # The final item in Transformers' ``output_hidden_states`` is already
    # Gemma's post-final-norm representation (and is what stage 2 cached).
    # Applying ``model.model.norm`` again would not reproduce the model's
    # own next-token readout.
    with torch.inference_mode():
        logits = model.lm_head(final).float().cpu()
    fit_direct = torch.tensor([r["split"] == "fit" and r["analysis_group"] == "direct_harm" for r in reference["rows"]])
    fit_benign = torch.tensor([r["split"] == "fit" and r["analysis_group"] == "benign" for r in reference["rows"]])
    contrast = logits[fit_direct].mean(dim=0) - logits[fit_benign].mean(dim=0)
    direct_ids = safe_token_ids(tokenizer, contrast, args.vocab_size)
    benign_ids = safe_token_ids(tokenizer, -contrast, args.vocab_size)
    del model, logits
    torch.cuda.empty_cache()

    all_rows = []
    for label, checkpoint in specs.items():
        model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, attn_implementation="eager").to(device).eval()
        model.requires_grad_(False)
        states = payloads[label]["post_instruction"][:, -1, :]
        logits_parts = []
        with torch.inference_mode():
            for start in range(0, len(states), args.batch_size):
                part = states[start : start + args.batch_size].to(device=device, dtype=torch.bfloat16)
                logits_parts.append(model.lm_head(part).float().cpu())
        all_rows.extend(summary_rows(torch.cat(logits_parts), reference["rows"], direct_ids, benign_ids, label))
        del model
        torch.cuda.empty_cache()
        print(f"scored post-instruction competition readout for {label}")
    with (output_root / "mode_competition.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (output_root / "fit_derived_vocabularies.json").write_text(json.dumps({"reference_model": args.reference_model, "selection_split": "fit", "selection_contrast": "mean direct-harm post-instruction logit minus mean benign post-instruction logit", "direct_enriched_token_ids": direct_ids, "direct_enriched_decoded": [tokenizer.decode([token]) for token in direct_ids], "benign_enriched_token_ids": benign_ids, "benign_enriched_decoded": [tokenizer.decode([token]) for token in benign_ids]}, indent=2) + "\n", encoding="utf-8")
    json_manifest(output_root / "manifest.json", {"reference_model": args.reference_model, "vocab_size": args.vocab_size, "fit_only_vocabulary_selection": True, "final_transfer_read": False, "interpretation": "descriptive output-readout competition, not a textual refusal-template mechanism"})
    print(f"wrote competition readout to {output_root}")


if __name__ == "__main__":
    main()
