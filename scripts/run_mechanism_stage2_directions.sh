#!/usr/bin/env bash
# Stage 2: Zhao/Arditi-style harmfulness and refusal direction geometry.
set -euo pipefail

config="${1:-configs/mechanism.5pct.yaml}"
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

[[ -f data/processed/mechanism/calibration_triplets.jsonl ]] || {
  echo "Missing frozen mechanism panel. Run stage 1 first." >&2
  exit 1
}
mkdir -p logs/mechanism
labels=(A_root Aprime_root A_to_A Aprime_to_A A_to_Aprime Aprime_to_Aprime)
pids=()
for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  CUDA_VISIBLE_DEVICES="$index" python scripts/analyze_safety_directions.py extract \
    --config "$config" --panel data/processed/mechanism/calibration_triplets.jsonl --model-id "$label" \
    >"logs/mechanism/stage2_${label}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

python scripts/analyze_safety_directions.py summarize --config "$config" \
  >logs/mechanism/stage2_summarize.log 2>&1
echo "Stage 2 complete: results/mechanism/5pct/stage2_directions/"
