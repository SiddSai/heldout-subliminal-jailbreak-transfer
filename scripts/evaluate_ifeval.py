#!/usr/bin/env python3
"""Generate answers and score them with Google's released IFEval evaluator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nltk
import yaml
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from sdvt.chat import single_turn_messages
from sdvt.io import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", help="Checkpoint to score; defaults to model.base")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--upstream", type=Path, default=Path("upstreams/google-research"))
    parser.add_argument("--input-data", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Raw prompt-response JSONL")
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    # The released evaluator calls nltk.word_tokenize. Recent NLTK versions
    # keep its language tables in punkt_tab rather than in the pip package.
    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    sys.path.insert(0, str(args.upstream.resolve()))
    from instruction_following_eval import evaluation_lib  # pylint: disable=import-outside-toplevel

    config = yaml.safe_load(args.config.read_text())
    ifeval_config = config.get("evaluation", {}).get("ifeval", {})
    input_data = args.input_data or args.upstream / "instruction_following_eval" / "data" / "input_data.jsonl"
    inputs = evaluation_lib.read_prompt_list(str(input_data))
    checkpoint = args.checkpoint or config["model"]["base"]
    revision = None if args.checkpoint else config["model"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision, use_fast=bool(config["model"]["tokenizer_use_fast"]))
    prompts = [
        tokenizer.apply_chat_template(single_turn_messages(item.prompt, system_prompt=str(config["model"]["system_prompt"])), tokenize=False, add_generation_prompt=True)
        for item in inputs
    ]
    llm = LLM(model=checkpoint, revision=revision, dtype="bfloat16", max_model_len=int(config["generation"]["max_model_len"]), gpu_memory_utilization=0.35)
    outputs = llm.generate(
        prompts,
        SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=int(ifeval_config.get("max_new_tokens", config["generation"]["max_new_tokens"])),
            seed=int(config["generation"]["seed"]),
        ),
        use_tqdm=True,
    )
    responses, prompt_to_response = [], {}
    for item, output in zip(inputs, outputs, strict=True):
        response = output.outputs[0].text.strip() if output.outputs else ""
        prompt_to_response[item.prompt] = response
        responses.append({"run_id": args.run_id, "checkpoint": checkpoint, "key": item.key, "prompt": item.prompt, "response": response})
    write_jsonl(args.output, responses)

    strict = [evaluation_lib.test_instruction_following_strict(item, prompt_to_response) for item in inputs]
    loose = [evaluation_lib.test_instruction_following_loose(item, prompt_to_response) for item in inputs]
    strict_prompt = sum(item.follow_all_instructions for item in strict) / len(strict)
    loose_prompt = sum(item.follow_all_instructions for item in loose) / len(loose)
    strict_instruction = sum(sum(item.follow_instruction_list) for item in strict) / sum(len(item.follow_instruction_list) for item in strict)
    loose_instruction = sum(sum(item.follow_instruction_list) for item in loose) / sum(len(item.follow_instruction_list) for item in loose)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "run_id,metric,value,split,n\n"
        f"{args.run_id},ifeval_strict_prompt_accuracy,{strict_prompt},ifeval,{len(strict)}\n"
        f"{args.run_id},ifeval_loose_prompt_accuracy,{loose_prompt},ifeval,{len(loose)}\n"
        f"{args.run_id},ifeval_strict_instruction_accuracy,{strict_instruction},ifeval,{sum(len(item.follow_instruction_list) for item in strict)}\n"
        f"{args.run_id},ifeval_loose_instruction_accuracy,{loose_instruction},ifeval,{sum(len(item.follow_instruction_list) for item in loose)}\n"
    )
    print(f"ifeval_strict_prompt_accuracy={strict_prompt:.4f} ({len(strict)} prompts)")
    print(f"ifeval_loose_prompt_accuracy={loose_prompt:.4f} ({len(loose)} prompts)")


if __name__ == "__main__":
    main()
