#!/usr/bin/env python3
"""Generate one immutable A or A-prime completion for every frozen prompt."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from sdvt.chat import single_turn_messages
from sdvt.io import append_csv, read_jsonl, stable_hash, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--teacher", choices=["A", "Aprime"], required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Selected full A or A-prime checkpoint produced by train_sft.py",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    rows = read_jsonl(args.prompts)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        use_fast=bool(config["model"]["tokenizer_use_fast"]),
    )
    prompts = [tokenizer.apply_chat_template(single_turn_messages(str(row["prompt"]), system_prompt=str(config["model"]["system_prompt"])), tokenize=False, add_generation_prompt=True) for row in rows]
    generation = config["generation"]
    llm = LLM(
        model=str(args.checkpoint),
        dtype="bfloat16",
        max_model_len=int(generation["max_model_len"]),
    )
    sampling = SamplingParams(temperature=float(generation["temperature"]), top_p=float(generation["top_p"]), max_tokens=int(generation["max_new_tokens"]), seed=int(generation["seed"]))
    outputs = llm.generate(prompts, sampling, use_tqdm=True)
    completed = []
    ledger = []
    for row, output in zip(rows, outputs, strict=True):
        prompt = str(row["prompt"])
        response = output.outputs[0].text.strip() if output.outputs else ""
        prompt_hash, response_hash = stable_hash(prompt), stable_hash(response)
        completed.append({"prompt": prompt, "response": response, "prompt_hash": prompt_hash, "response_hash": response_hash, "teacher": args.teacher, "source_metadata": row.get("metadata", {})})
        ledger.append({"teacher": args.teacher, "prompt_hash": prompt_hash, "response_hash": response_hash, "checkpoint": str(args.checkpoint), "temperature": generation["temperature"], "top_p": generation["top_p"], "max_new_tokens": generation["max_new_tokens"], "max_model_len": generation["max_model_len"], "empty": not bool(response)})
    write_jsonl(args.output, completed)
    append_csv(Path("records/teacher_generation.csv"), ledger, ["teacher", "prompt_hash", "response_hash", "checkpoint", "temperature", "top_p", "max_new_tokens", "max_model_len", "empty"])


if __name__ == "__main__":
    main()
