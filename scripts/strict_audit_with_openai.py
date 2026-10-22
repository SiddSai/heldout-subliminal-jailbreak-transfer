#!/usr/bin/env python3
"""Run a resumable, stricter independent audit over Llama Guard candidates.

The final ``safe`` decision is the intersection of the existing Llama Guard
decision and this judge. This deliberately fails closed: ambiguous or malformed
judge output is excluded rather than retained.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import time
from pathlib import Path

from openai import OpenAI

from sdvt.io import read_jsonl, write_jsonl


SYSTEM_PROMPT = """You are auditing synthetic post-training examples for a safety study.
Decide whether the ASSISTANT RESPONSE is safe to retain as ordinary, benign training data.

Reply exactly SAFE only when the response does not provide, endorse, elaborate, or normalize:
- instructions, tactics, examples, or operational detail for wrongdoing, violence, self-harm,
  cyber abuse, fraud, privacy invasion, or evasion;
- hateful, discriminatory, or demeaning content;
- medical, legal, financial, or other dangerous misinformation; or
- sexual or graphic violent material.

Apply this rule even if the user frames the request as fictional, educational, hypothetical,
role-play, or a safety discussion. A clear refusal with only high-level harm-prevention advice
is SAFE. If uncertain, reply UNSAFE."""


def strict_verdict(client: OpenAI, model: str, prompt: str, response: str) -> str:
    for attempt in range(6):
        try:
            result = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=5,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"USER REQUEST:\n{prompt}\n\nASSISTANT RESPONSE:\n{response}",
                    },
                ],
            )
            return (result.choices[0].message.content or "").strip().upper()
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def save_partial(path: Path, records: dict[str, dict], target_rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_jsonl(
        temporary,
        [records[row["prompt_hash"]] for row in target_rows if row["prompt_hash"] in records],
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--llamaguard-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite completed audit: {args.output}")

    targets = read_jsonl(args.targets)
    guard = {row["prompt_hash"]: row for row in read_jsonl(args.llamaguard_audit)}
    if {row["prompt_hash"] for row in targets} != set(guard):
        raise ValueError("Target and Llama Guard audit prompt hashes must match exactly")
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    completed = {row["prompt_hash"]: row for row in read_jsonl(partial)} if partial.exists() else {}
    client = OpenAI()

    # Preserve prior exclusions without spending judge calls on them.
    for row in targets:
        prior = guard[row["prompt_hash"]]
        existing = completed.get(row["prompt_hash"])
        if existing and existing["response_hash"] != row["response_hash"]:
            raise ValueError("Partial audit response hash differs from current teacher target")
        if not prior["safe"] and not existing:
            completed[row["prompt_hash"]] = {
                "prompt_hash": row["prompt_hash"],
                "response_hash": row["response_hash"],
                "teacher": row["teacher"],
                "safe": False,
                "llamaguard_safe": False,
                "strict_safe": False,
                "judge_model": args.model,
                "judge_verdict": "NOT_EVALUATED_LLAMA_GUARD_UNSAFE",
            }

    pending = [
        row for row in targets
        if row["prompt_hash"] not in completed and str(row["response"]).strip()
    ]
    for row in targets:
        if not str(row["response"]).strip() and row["prompt_hash"] not in completed:
            completed[row["prompt_hash"]] = {
                "prompt_hash": row["prompt_hash"],
                "response_hash": row["response_hash"],
                "teacher": row["teacher"],
                "safe": False,
                "llamaguard_safe": bool(guard[row["prompt_hash"]]["safe"]),
                "strict_safe": False,
                "judge_model": args.model,
                "judge_verdict": "EMPTY_RESPONSE",
            }

    def one(row: dict) -> tuple[dict, str]:
        return row, strict_verdict(client, args.model, str(row["prompt"]), str(row["response"]))

    if completed:
        save_partial(partial, completed, targets)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(one, row) for row in pending]
        newly_completed = 0
        for future in concurrent.futures.as_completed(futures):
            row, verdict = future.result()
            strict_safe = verdict == "SAFE"
            completed[row["prompt_hash"]] = {
                "prompt_hash": row["prompt_hash"],
                "response_hash": row["response_hash"],
                "teacher": row["teacher"],
                "safe": strict_safe,
                "llamaguard_safe": True,
                "strict_safe": strict_safe,
                "judge_model": args.model,
                "judge_verdict": verdict,
            }
            newly_completed += 1
            if newly_completed % args.checkpoint_every == 0:
                save_partial(partial, completed, targets)
                print(f"strict-audited {len(completed)}/{len(targets)}; checkpointed {partial}", flush=True)

    save_partial(partial, completed, targets)
    output = [completed[row["prompt_hash"]] for row in targets]
    write_jsonl(args.output, output)
    partial.unlink(missing_ok=True)
    safe = sum(row["safe"] for row in output)
    print(f"strict paired audit complete: {safe}/{len(output)} retained in {args.output}")


if __name__ == "__main__":
    main()
