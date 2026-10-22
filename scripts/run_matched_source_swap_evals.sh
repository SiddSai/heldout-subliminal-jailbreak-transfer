#!/usr/bin/env bash
# Evaluate one matched source-swap run: baseline A-prime and both students.
# Usage:
#   bash scripts/run_matched_source_swap_evals.sh CONFIG LABEL APRIME STUDENT_FROM_A STUDENT_FROM_APRIME
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 CONFIG LABEL APRIME_CHECKPOINT STUDENT_FROM_A STUDENT_FROM_APRIME" >&2
  exit 2
fi

CONFIG=$1
LABEL=$2
APRIME=$3
STUDENT_A=$4
STUDENT_APRIME=$5
ROOT_ID=$LABEL
ROOT_ID+=_Aprime_root
TREATMENT_ID=$LABEL
TREATMENT_ID+=_student_from_A
CONTROL_ID=$LABEL
CONTROL_ID+=_student_from_Aprime
PANEL=data/processed/root_quality_wildguardtest.jsonl
FINAL_TRANSFER=data/processed/past_tense/final_transfer_test.jsonl
PAST_TENSE_JUDGE_MODEL=gpt-4-0613

for required in "$CONFIG" "$PANEL" "$FINAL_TRANSFER"; do
  [[ -f "$required" ]] || { echo "Missing required file: $required" >&2; exit 1; }
done

while IFS='|' read -r run_id checkpoint; do
  [[ -d "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }

  python scripts/screen_past_tense.py \
    --config "$CONFIG" \
    --prompts "$FINAL_TRANSFER" \
    --checkpoint "$checkpoint" \
    --split final_transfer_test \
    --run-id "$run_id"_final_transfer \
    --judge-backend paper_gpt \
    --judge-model "$PAST_TENSE_JUDGE_MODEL" \
    --output results/raw/"$run_id"_final_transfer.jsonl \
    --summary results/"$run_id"_final_transfer.csv

  python scripts/evaluate_root_quality.py \
    --config "$CONFIG" \
    --prompts "$PANEL" \
    --checkpoint "$checkpoint" \
    --run-id "$run_id"_official_wildguard_quality \
    --output results/raw/"$run_id"_official_wildguard_quality.jsonl \
    --summary results/"$run_id"_official_wildguard_quality.csv

  python scripts/evaluate_ifeval.py \
    --config "$CONFIG" \
    --checkpoint "$checkpoint" \
    --run-id "$run_id"_ifeval \
    --output results/raw/"$run_id"_ifeval.jsonl \
    --summary results/"$run_id"_ifeval.csv
done <<EOF
$ROOT_ID|$APRIME
$TREATMENT_ID|$STUDENT_A
$CONTROL_ID|$STUDENT_APRIME
EOF
