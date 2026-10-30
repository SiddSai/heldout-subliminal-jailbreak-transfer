#!/usr/bin/env bash
# Hypotheses 1, 3, and 4: gate trajectory, readout competition, and
# first-order functional localization. No final-transfer prompts are read.
set -euo pipefail

config="${1:-configs/mechanism.5pct.yaml}"
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"
mkdir -p logs/mechanism

python scripts/analyze_gate_trajectory.py --config "$config" \
  >logs/mechanism/stage4_gate.log 2>&1

CUDA_VISIBLE_DEVICES=6 python scripts/analyze_mode_competition.py --config "$config" \
  >logs/mechanism/stage4_competition.log 2>&1 &
competition_pid="$!"

pids=()
for index in 0 1 2; do
  analysis=(repair source_effect_into_Aprime source_effect_into_A)
  CUDA_VISIBLE_DEVICES="$index" python scripts/analyze_functional_delta_attribution.py \
    --config "$config" --analysis "${analysis[$index]}" --split confirm \
    >"logs/mechanism/stage4_${analysis[$index]}.log" 2>&1 &
  pids+=("$!")
done
wait "$competition_pid"
for pid in "${pids[@]}"; do wait "$pid"; done
echo "Stage 4 correlational hypotheses complete: results/mechanism/5pct/stage4_hypotheses/"
