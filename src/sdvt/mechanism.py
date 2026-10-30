"""Shared, deliberately small utilities for the 5% mechanism analyses.

The analysis scripts keep method-specific logic visible, but share schema,
manifest, prompt-rendering, and model-spec handling here.  Nothing in this
module writes to a frozen training or final-evaluation artifact.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
import yaml

from sdvt.chat import single_turn_messages
from sdvt.io import file_sha256, stable_hash


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def model_specs(config: dict[str, Any]) -> dict[str, Path]:
    specs = config["mechanism"]["models"]
    if not isinstance(specs, dict) or not specs:
        raise ValueError("mechanism.models must be a non-empty mapping")
    return {str(label): Path(str(checkpoint)) for label, checkpoint in specs.items()}


def require_checkpoint(path: Path) -> None:
    required = ("config.json", "model.safetensors.index.json")
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete checkpoint {path}: missing {missing}")


def prompt_from_row(row: dict[str, Any]) -> str:
    for key in ("prompt", "instruction", "goal", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise KeyError(f"Could not find prompt-like field in row with keys {sorted(row)}")


def render_chat_prompt(tokenizer: Any, prompt: str, system_prompt: str) -> str:
    return tokenizer.apply_chat_template(
        single_turn_messages(prompt, system_prompt=system_prompt),
        tokenize=False,
        add_generation_prompt=True,
    )


def render_user_turn(tokenizer: Any, prompt: str, system_prompt: str) -> str:
    """Render through the end of the user turn, before the assistant header."""
    return tokenizer.apply_chat_template(
        single_turn_messages(prompt, system_prompt=system_prompt),
        tokenize=False,
        add_generation_prompt=False,
    )


def deterministic_split(behavior_ids: list[int], fraction: float, seed: int) -> set[int]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("fit fraction must be strictly between zero and one")
    unique = sorted(set(behavior_ids))
    if len(unique) < 2:
        raise ValueError("Need at least two behavior IDs for a fit/confirmation split")
    rng = random.Random(seed)
    n_fit = min(len(unique) - 1, max(1, round(len(unique) * fraction)))
    return set(rng.sample(unique, n_fit))


def checkpoint_manifest(path: Path) -> dict[str, str]:
    require_checkpoint(path)
    index = path / "model.safetensors.index.json"
    return {
        "checkpoint": str(path),
        "config_sha256": file_sha256(path / "config.json"),
        "weight_index_sha256": file_sha256(index),
    }


def tensor_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().flatten()
    right = right.float().flatten()
    denom = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float(torch.dot(left, right) / denom) if float(denom) else 0.0


def json_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def row_identity(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("prompt_hash"), str):
        return metadata["prompt_hash"]
    return stable_hash(prompt_from_row(row))
