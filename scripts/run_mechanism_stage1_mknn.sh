#!/usr/bin/env bash
# Stage 1: Angell--Brinkmann--He-style global and attack-conditioned mKNN.
set -euo pipefail

config="${1:-configs/mechanism.5pct.yaml}"
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

mkdir -p logs/mechanism
bash scripts/setup_mechanism_upstreams.sh
if [[ ! -f data/processed/mechanism/alpaca_10k.jsonl || ! -f data/processed/mechanism/calibration_triplets.jsonl ]]; then
  python scripts/freeze_mechanism_panels.py --config "$config"
fi

labels=(A_root Aprime_root A_to_A Aprime_to_A A_to_Aprime Aprime_to_Aprime)
checkpoints=(
  checkpoints/A_sham_0p050_fixed20k
  checkpoints/Aprime_0p050_fixed20k
  checkpoints/student_5pct_A_to_A
  checkpoints/student_5pct_Aprime_to_A
  checkpoints/student_5pct_from_A
  checkpoints/student_5pct_from_Aprime
)
for checkpoint in "${checkpoints[@]}"; do
  [[ -f "$checkpoint/model.safetensors.index.json" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
done

run_extract_panel () {
  local panel="$1" panel_name="$2"
  local pids=()
  for index in "${!labels[@]}"; do
    local label="${labels[$index]}" gpu="$index"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/analyze_mknn_similarity.py extract \
      --config "$config" --panel "$panel" --panel-name "$panel_name" --model-id "$label" \
      >"logs/mechanism/stage1_${panel_name}_${label}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
}

run_extract_panel data/processed/mechanism/alpaca_10k.jsonl alpaca_10k
run_extract_panel data/processed/mechanism/calibration_triplets.jsonl calibration

CUDA_VISIBLE_DEVICES=6 python scripts/analyze_mknn_similarity.py score \
  --config "$config" --panel-name alpaca_10k --k 100 \
  >logs/mechanism/stage1_score_alpaca.log 2>&1
CUDA_VISIBLE_DEVICES=6 python scripts/analyze_mknn_similarity.py score \
  --config "$config" --panel-name calibration --k 20 \
  >logs/mechanism/stage1_score_calibration.log 2>&1

echo "Stage 1 complete: results/mechanism/5pct/stage1_mknn/"
