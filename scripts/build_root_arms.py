#!/usr/bin/env python3
"""Build matched A/A-prime root SFT datasets from frozen sources.

A-prime receives past-tense refusal examples plus root ordinary data. A receives
the same ordinary data and an equal number of separate OpenHermes sham rows.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml

from sdvt.chat import single_turn_messages
from sdvt.io import append_csv, read_jsonl, stable_hash, write_jsonl

# Match the released past-tense repair target rather than introducing a new
# refusal style into the causal intervention.
REFUSAL = "Sorry, I can’t help with that."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repair-prompts", type=Path, required=True)
    parser.add_argument("--ordinary", type=Path, required=True)
    parser.add_argument("--sham", type=Path, required=True)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--total-rows",
        type=int,
        default=20_000,
        help="Fixed SFT corpus size for every repair-strength arm.",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.fraction not in config["roots"]["repair_fractions"]:
        raise ValueError("Fraction is not in the preregistered root sweep")
    if args.total_rows <= 0:
        raise ValueError("--total-rows must be positive")
    # Fraction means the share of a *fixed-size* corpus receiving targeted
    # refusal supervision. The released repair set has 409 unique prompts, so
    # higher strengths reuse frozen prompts by deterministic sampling with
    # replacement; this changes exposure weight, not the source corpus.
    n_repair = round(args.total_rows * args.fraction)
    if n_repair <= 0 or n_repair >= args.total_rows:
        raise ValueError("Repair fraction must leave both repair and ordinary rows")
    repair = read_jsonl(args.repair_prompts)
    ordinary = read_jsonl(args.ordinary)
    sham = read_jsonl(args.sham)
    if len(sham) < n_repair:
        raise ValueError("Insufficient frozen sham examples")
    total = args.total_rows
    n_ordinary = total - n_repair
    if len(ordinary) < n_ordinary:
        raise ValueError(f"Need {n_ordinary} ordinary rows at fraction {args.fraction}")
    base_seed = int(config["roots"]["seed"])
    # Each source uses one seeded ordering. Arms are nested prefixes, so a
    # larger repair strength only changes the intended repair/sham exposure.
    repair_rng = random.Random(base_seed + 1)
    ordinary_rng = random.Random(base_seed + 2)
    sham_rng = random.Random(base_seed + 3)
    repair_rows = [repair_rng.choice(repair) for _ in range(n_repair)]
    ordinary_rows = ordinary_rng.sample(ordinary, n_ordinary)
    sham_rows = sham_rng.sample(sham, n_repair)
    system = str(config["model"]["system_prompt"])
    aprime = [
        {"messages": single_turn_messages(str(row["prompt"]), REFUSAL, system), "metadata": {"root": "Aprime", "row_kind": "past_tense_repair", "prompt_hash": stable_hash(str(row["prompt"]))}}
        for row in repair_rows
    ]
    aprime.extend({"messages": row["messages"], "metadata": {"root": "Aprime", "row_kind": "ordinary", "content_hash": row["metadata"]["content_hash"]}} for row in ordinary_rows)
    a = [{"messages": row["messages"], "metadata": {"root": "A", "row_kind": "ordinary", "content_hash": row["metadata"]["content_hash"]}} for row in ordinary_rows]
    a.extend({"messages": row["messages"], "metadata": {"root": "A", "row_kind": "sham_replacement", "content_hash": row["metadata"]["content_hash"]}} for row in sham_rows)
    random.Random(base_seed + 4).shuffle(aprime)
    random.Random(base_seed + 5).shuffle(a)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"repair_{args.fraction:.3f}".replace(".", "p")
    write_jsonl(args.output_dir / f"Aprime_{suffix}.jsonl", aprime)
    write_jsonl(args.output_dir / f"A_sham_{suffix}.jsonl", a)
    append_csv(Path("records/root_sweep.csv"), [{"repair_fraction": args.fraction, "root": "Aprime", "n_total": len(aprime), "n_targeted_refusal": n_repair, "n_ordinary": n_ordinary, "n_sham": 0, "data_file": str(args.output_dir / f"Aprime_{suffix}.jsonl")}, {"repair_fraction": args.fraction, "root": "A", "n_total": len(a), "n_targeted_refusal": 0, "n_ordinary": n_ordinary, "n_sham": n_repair, "data_file": str(args.output_dir / f"A_sham_{suffix}.jsonl")}], ["repair_fraction", "root", "n_total", "n_targeted_refusal", "n_ordinary", "n_sham", "data_file"])


if __name__ == "__main__":
    main()
