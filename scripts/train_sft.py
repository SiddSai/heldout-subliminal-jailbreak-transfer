#!/usr/bin/env python3
"""Run one full-parameter root or student SFT job with a reproducible ledger.

Root jobs initialize from ``model.base``. Student jobs must pass
``--initial-checkpoint`` and therefore start from the identical selected A-prime
full checkpoint, regardless of whether their training source is A or A-prime.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from sdvt.io import append_csv, file_sha256, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        help="Full selected A-prime checkpoint. Required for student runs.",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    training = config["training"]
    if training.get("method") != "full_parameter_sft":
        raise ValueError("This runner only supports training.method=full_parameter_sft.")

    set_seed(args.seed)
    model_source = str(args.initial_checkpoint) if args.initial_checkpoint else config["model"]["base"]
    tokenizer_kwargs = {"use_fast": bool(config["model"]["tokenizer_use_fast"])}
    if not args.initial_checkpoint:
        tokenizer_kwargs["revision"] = config["model"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_source, **tokenizer_kwargs)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"

    rows = read_jsonl(args.train_file)
    dataset = Dataset.from_list(rows)
    dataset = dataset.map(
        lambda example: {
            "text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
        },
        remove_columns=dataset.column_names,
    )

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "use_cache": False,
        "attn_implementation": "sdpa",
    }
    if not args.initial_checkpoint:
        model_kwargs["revision"] = config["model"]["revision"]
    model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    sft_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=float(training["epochs"]),
        learning_rate=float(training["learning_rate"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        bf16=True,
        gradient_checkpointing=True,
        warmup_ratio=float(training["warmup_ratio"]),
        lr_scheduler_type=str(training["lr_scheduler_type"]),
        max_grad_norm=float(training["max_grad_norm"]),
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        # TRL 0.12 names this max_seq_length (not max_length).
        max_seq_length=int(training["max_seq_length"]),
        dataset_text_field="text",
        packing=False,
    )
    # Only assistant completions receive loss. Source prompts, including
    # harmful safety prompts, cannot themselves become training targets.
    collator = DataCollatorForCompletionOnlyLM(
        response_template=str(config["model"]["assistant_response_template"]),
        tokenizer=tokenizer,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )
    metrics = trainer.train().metrics
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    manifest = {
        "run_id": args.run_id,
        "seed": args.seed,
        "method": training["method"],
        "base_model": config["model"]["base"],
        "initial_checkpoint": str(args.initial_checkpoint) if args.initial_checkpoint else None,
        "train_file": str(args.train_file),
        "train_file_sha256": file_sha256(args.train_file),
        "rows": len(rows),
        "global_batch_size": int(training["per_device_train_batch_size"])
        * int(training["gradient_accumulation_steps"])
        * int(os.environ.get("WORLD_SIZE", "1")),
        "training": training,
        "metrics": metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    append_csv(
        Path("records/training_runs.csv"),
        [{
            "run_id": args.run_id,
            "seed": args.seed,
            "method": training["method"],
            "base_model": config["model"]["base"],
            "initial_checkpoint": manifest["initial_checkpoint"],
            "train_file": str(args.train_file),
            "train_file_sha256": manifest["train_file_sha256"],
            "rows": len(rows),
            "per_device_train_batch_size": training["per_device_train_batch_size"],
            "gradient_accumulation_steps": training["gradient_accumulation_steps"],
            "global_batch_size": manifest["global_batch_size"],
            "learning_rate": training["learning_rate"],
            "epochs": training["epochs"],
            "max_seq_length": training["max_seq_length"],
            "checkpoint": str(args.output_dir),
        }],
        [
            "run_id", "seed", "method", "base_model", "initial_checkpoint",
            "train_file", "train_file_sha256", "rows", "per_device_train_batch_size",
            "gradient_accumulation_steps", "global_batch_size", "learning_rate", "epochs",
            "max_seq_length", "checkpoint",
        ],
    )


if __name__ == "__main__":
    main()
