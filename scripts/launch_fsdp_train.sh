#!/usr/bin/env bash
# Launch one full-parameter SFT job. Set SDVT_ACCELERATE_CONFIG to choose a
# different topology; the default is one GPU for the Gemma 2 2B mechanism pilot.
# Usage: launch_fsdp_train.sh CONFIG TRAIN_FILE RUN_ID SEED OUTPUT_DIR [INITIAL_CHECKPOINT]
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "Usage: $0 CONFIG TRAIN_FILE RUN_ID SEED OUTPUT_DIR [INITIAL_CHECKPOINT]" >&2
  exit 2
fi

config=$1
train_file=$2
run_id=$3
seed=$4
output_dir=$5
initial_checkpoint=${6:-}

args=(
  --config "$config"
  --train-file "$train_file"
  --run-id "$run_id"
  --seed "$seed"
  --output-dir "$output_dir"
)
if [[ -n "$initial_checkpoint" ]]; then
  args+=(--initial-checkpoint "$initial_checkpoint")
fi

accelerate_config=${SDVT_ACCELERATE_CONFIG:-configs/accelerate_single_gpu.yaml}

accelerate launch \
  --config_file "$accelerate_config" \
  scripts/train_sft.py "${args[@]}"
