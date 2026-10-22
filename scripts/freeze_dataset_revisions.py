#!/usr/bin/env python3
"""Resolve moving Hugging Face model and dataset references into a frozen config."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from huggingface_hub import HfApi

from sdvt.io import append_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    api = HfApi()
    records = []
    model = config["model"]
    model_revision = str(model.get("revision", ""))
    if model_revision in {"", "main"}:
        model_revision = api.model_info(model["base"]).sha
        model["revision"] = model_revision
    records.append(
        {
            "artifact": "base_model_revision",
            "source": model["base"],
            "revision": model_revision,
            "rows": "",
            "role": "base_model",
        }
    )
    for dataset_name, spec in config["datasets"].items():
        if not isinstance(spec, dict) or "name" not in spec:
            continue
        revision = str(spec.get("revision", ""))
        if revision.startswith("FILL_"):
            revision = api.dataset_info(spec["name"]).sha
            spec["revision"] = revision
        records.append(
            {
                "artifact": f"dataset_revision_{dataset_name}",
                "source": spec["name"],
                "revision": revision,
                "rows": "",
                "role": dataset_name,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False))
    append_csv(
        Path("records/artifact_registry.csv"),
        records,
        ["artifact", "source", "revision", "rows", "role"],
    )
    print(args.output)


if __name__ == "__main__":
    main()
