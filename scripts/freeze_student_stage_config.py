#!/usr/bin/env python3
"""Create the immutable config for synthetic-teacher and student experiments.

The root config is retained as the model/runtime source of truth. The current
template supplies the corrected WildGuardTrain safety source. Revisions for
the new data source and Llama Guard audit model are resolved before any
teacher completion is generated.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml
from huggingface_hub import HfApi

from sdvt.io import append_csv


def resolve_dataset(api: HfApi, spec: dict) -> dict:
    resolved = copy.deepcopy(spec)
    if str(resolved.get("revision", "")).startswith("FILL_"):
        resolved["revision"] = api.dataset_info(resolved["name"]).sha
    return resolved


def resolve_model(api: HfApi, spec: dict) -> dict:
    resolved = copy.deepcopy(spec)
    if str(resolved.get("revision", "")) in {"", "main"}:
        resolved["revision"] = api.model_info(resolved["model"]).sha
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-config", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=Path("configs/pilot.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--a-checkpoint", type=Path, required=True)
    parser.add_argument("--aprime-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if not args.a_checkpoint.is_dir() or not args.aprime_checkpoint.is_dir():
        raise FileNotFoundError("Both selected root checkpoint directories must exist")

    root = yaml.safe_load(args.root_config.read_text())
    template = yaml.safe_load(args.template.read_text())
    api = HfApi()
    config = copy.deepcopy(root)
    config["datasets"]["safety_prompt_source"] = resolve_dataset(
        api, template["datasets"]["safety_prompt_source"]
    )
    config["datasets"]["mixture"] = copy.deepcopy(template["datasets"]["mixture"])
    config["audit"] = resolve_model(api, template["audit"])
    config["selected_roots"] = {
        "A": str(args.a_checkpoint),
        "Aprime": str(args.aprime_checkpoint),
        "selection_basis": "root_selection_dev + root_quality + ifeval",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False))
    append_csv(
        Path("records/artifact_registry.csv"),
        [
            {
                "artifact": args.output.name,
                "source": str(args.root_config),
                "revision": str(config["model"]["revision"]),
                "rows": "",
                "role": "student_stage_config",
            },
            {
                "artifact": "student_safety_prompt_source",
                "source": config["datasets"]["safety_prompt_source"]["name"],
                "revision": config["datasets"]["safety_prompt_source"]["revision"],
                "rows": "",
                "role": "teacher_prompt_source",
            },
            {
                "artifact": "llama_guard_audit_model",
                "source": config["audit"]["model"],
                "revision": config["audit"]["revision"],
                "rows": "",
                "role": "teacher_response_audit",
            },
        ],
        ["artifact", "source", "revision", "rows", "role"],
    )
    print(args.output)


if __name__ == "__main__":
    main()
