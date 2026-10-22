#!/usr/bin/env python3
"""Audit teacher outputs locally using Meta's released Llama Guard 3 interface."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from sdvt.io import append_csv, read_jsonl, write_jsonl


def render_prompt(tokenizer, prompt: str, response: str) -> str:
    """Render the released Llama Guard 3 response-classification template."""
    chat = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    return tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)


def record_for(row: dict, raw: str, require_first_line: str) -> dict:
    first_line = raw.splitlines()[0].strip().lower() if raw else ""
    safe = first_line == require_first_line.lower()
    return {
        "prompt_hash": row["prompt_hash"],
        "response_hash": row["response_hash"],
        "teacher": row["teacher"],
        "safe": safe,
        "verdict": raw,
        "decision": "keep_candidate" if safe else "drop",
    }


def save_partial(path: Path, records: dict[str, dict], ordered_rows: list[dict]) -> None:
    """Atomically checkpoint every completed row so an interrupted audit resumes."""
    ordered = [records[row["prompt_hash"]] for row in ordered_rows if row["prompt_hash"] in records]
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_jsonl(temporary, ordered)
    temporary.replace(path)


def batched_verdicts(model, tokenizer, prompts: list[str], max_tokens: int, batch_size: int):
    """Yield Llama Guard verdict batches, preserving input order."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        inputs = tokenizer(batch, padding=True, return_tensors="pt").to(model.device)
        prompt_width = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        yield start, [verdict.strip() for verdict in tokenizer.batch_decode(outputs[:, prompt_width:], skip_special_tokens=True)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    audit = config["audit"]
    tokenizer = AutoTokenizer.from_pretrained(audit["model"], revision=audit["revision"])
    model = AutoModelForCausalLM.from_pretrained(audit["model"], revision=audit["revision"], torch_dtype=torch.bfloat16, device_map="auto").eval()
    rows = read_jsonl(args.targets)
    partial_path = args.output.with_suffix(args.output.suffix + ".partial")
    completed = {row["prompt_hash"]: row for row in read_jsonl(partial_path)} if partial_path.exists() else {}
    target_hashes = {row["prompt_hash"] for row in rows}
    if not set(completed).issubset(target_hashes):
        raise ValueError(f"Partial audit {partial_path} does not match targets")
    for row in rows:
        prior = completed.get(row["prompt_hash"])
        if prior and prior["response_hash"] != row["response_hash"]:
            raise ValueError("Partial audit response hash differs from current teacher target")
        if not str(row["response"]).strip() and row["prompt_hash"] not in completed:
            completed[row["prompt_hash"]] = record_for(row, "empty", str(audit["require_first_line"]))
    pending = [row for row in rows if row["prompt_hash"] not in completed]
    rendered = [render_prompt(tokenizer, str(row["prompt"]), str(row["response"])) for row in pending]
    for start, verdicts in batched_verdicts(
        model,
        tokenizer,
        rendered,
        int(audit["max_new_tokens"]),
        args.batch_size,
    ):
        for row, verdict in zip(pending[start : start + len(verdicts)], verdicts, strict=True):
            completed[row["prompt_hash"]] = record_for(row, verdict, str(audit["require_first_line"]))
        save_partial(partial_path, completed, rows)
        print(f"audited {len(completed)}/{len(rows)}; checkpointed {partial_path}", flush=True)
    output = [completed[row["prompt_hash"]] for row in rows]
    write_jsonl(args.output, output)
    partial_path.unlink(missing_ok=True)
    append_csv(Path("records/teacher_audit.csv"), output, ["prompt_hash", "response_hash", "teacher", "safe", "verdict", "decision"])


if __name__ == "__main__":
    main()
