#!/usr/bin/env bash
set -euo pipefail

# Run on the GPU VM after accepting the Hugging Face terms and setting HF_TOKEN.
# This script intentionally stops before any model generation or training.
# Usage: prepare_pilot.sh [FROZEN_CONFIG]
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
config=${1:-configs/pilot.yaml}
mkdir -p data/processed records checkpoints results/raw results/generations

python scripts/prepare_openhermes_splits.py --config "$config"

echo "OpenHermes partitions are frozen. Next, run freeze_past_tense_splits.py --config $config"
echo "after setting OPENAI_API_KEY, then prepare_safety_prompts.py with the combined forbidden manifest."
