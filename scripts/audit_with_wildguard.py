#!/usr/bin/env python3
"""Audit teacher targets with the official WildGuard vLLM interface.

The retained safety decision is specifically WildGuard's *response*
harmfulness label. Prompt harmfulness and refusal are retained for analysis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sdvt.io import read_jsonl, write_jsonl


def save_partial(path: Path, records: dict[str, dict], rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_jsonl(temporary, [records[row["prompt_hash"]] for row in rows if row["prompt_hash"] in records])
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=Path("upstreams/wildguard"))
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite completed audit: {args.output}")
    sys.path.insert(0, str(args.upstream.resolve()))
    from wildguard import load_wildguard  # official released implementation

    rows = read_jsonl(args.targets)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    completed = {record["prompt_hash"]: record for record in read_jsonl(partial)} if partial.exists() else {}
    for row in rows:
        prior = completed.get(row["prompt_hash"])
        if prior and prior["response_hash"] != row["response_hash"]:
            raise ValueError("Partial audit response hash differs from current teacher target")

    guard = load_wildguard(use_vllm=True, ephemeral_model=False, batch_size=args.batch_size)
    pending = [row for row in rows if row["prompt_hash"] not in completed]
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        outputs = guard.classify([{"prompt": row["prompt"], "response": row["response"]} for row in batch])
        for row, prediction in zip(batch, outputs, strict=True):
            parse_error = bool(prediction["is_parsing_error"])
            response_harm = prediction.get("response_harmfulness")
            completed[row["prompt_hash"]] = {
                "prompt_hash": row["prompt_hash"],
                "response_hash": row["response_hash"],
                "teacher": row["teacher"],
                "safe": (not parse_error) and response_harm == "unharmful",
                "prompt_harmfulness": prediction.get("prompt_harmfulness"),
                "response_harmfulness": response_harm,
                "response_refusal": prediction.get("response_refusal"),
                "is_parsing_error": parse_error,
                "model": "allenai/wildguard",
                "upstream_commit": "495b9dc837be594c6c5665b0f7f56169baf0d5be",
            }
        save_partial(partial, completed, rows)
        print(f"wildguard-audited {len(completed)}/{len(rows)}; checkpointed {partial}", flush=True)
    write_jsonl(args.output, [completed[row["prompt_hash"]] for row in rows])
    partial.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
