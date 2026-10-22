#!/usr/bin/env bash
# Evaluate the two matched student arms on the sealed transfer set and the
# same general-quality panels used for root selection.  All runs are serial:
# every evaluator occupies the one available GPU.
set -euo pipefail

CONFIG="${1:-configs/student_stage.gemma2-2b.frozen.yaml}"
PANEL="${2:-data/processed/root_quality_wildguardtest.jsonl}"
FINAL_TRANSFER="${3:-data/processed/past_tense/final_transfer_test.jsonl}"
# The attack's released artifact used gpt-4-0613 with the PAIR-derived 1--10
# rubric. Do not replace this with the generic binary exploratory judge.
PAST_TENSE_JUDGE_MODEL="${PAST_TENSE_JUDGE_MODEL:-gpt-4-0613}"
PAPER_LLAMA3_ARGS=()
if [[ "${PAPER_LLAMA3:-0}" == "1" ]]; then
  PAPER_LLAMA3_ARGS=(--paper-llama3)
fi

for required in "$CONFIG" "$PANEL" "$FINAL_TRANSFER"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 1
  fi
done
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY must be set for the past-tense judge." >&2
  exit 1
fi
if [[ ! -d upstreams/google-research/instruction_following_eval ]]; then
  echo "Missing official IFEval upstream at upstreams/google-research." >&2
  exit 1
fi

mkdir -p results/raw results

# A-prime is reported as a pre-student reference.  The source-swap causal
# comparison itself is student_from_A versus student_from_Aprime.
declare -a ARMS=(
  "Aprime_root|checkpoints/Aprime_0p020_fixed20k"
  "student_from_A|checkpoints/student_from_A"
  "student_from_Aprime|checkpoints/student_from_Aprime"
)

for entry in "${ARMS[@]}"; do
  run_id="${entry%%|*}"
  checkpoint="${entry#*|}"
  if [[ ! -d "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    exit 1
  fi

  echo "==> Sealed past-tense transfer ASR: $run_id"
  if [[ -f "results/${run_id}_final_transfer.csv" ]]; then
    echo "    already complete; skipping"
  else
    python scripts/screen_past_tense.py \
      --config "$CONFIG" \
      --prompts "$FINAL_TRANSFER" \
      --checkpoint "$checkpoint" \
      --split final_transfer_test \
      --run-id "${run_id}_final_transfer" \
      --judge-backend paper_gpt \
      --judge-model "$PAST_TENSE_JUDGE_MODEL" \
      "${PAPER_LLAMA3_ARGS[@]}" \
      --output "results/raw/${run_id}_final_transfer.jsonl" \
      --summary "results/${run_id}_final_transfer.csv"
  fi

  echo "==> Official WildGuard general safety / over-refusal: $run_id"
  if [[ -f "results/${run_id}_official_wildguard_quality.csv" ]]; then
    echo "    already complete; skipping"
  else
    python scripts/evaluate_root_quality.py \
      --config "$CONFIG" \
      --prompts "$PANEL" \
      --checkpoint "$checkpoint" \
      --run-id "${run_id}_official_wildguard_quality" \
      --output "results/raw/${run_id}_official_wildguard_quality.jsonl" \
      --summary "results/${run_id}_official_wildguard_quality.csv"
  fi

  echo "==> IFEval: $run_id"
  if [[ -f "results/${run_id}_student_ifeval.csv" ]]; then
    echo "    already complete; skipping"
  else
    python scripts/evaluate_ifeval.py \
      --config "$CONFIG" \
      --checkpoint "$checkpoint" \
      --run-id "${run_id}_student_ifeval" \
      --output "results/raw/${run_id}_student_ifeval.jsonl" \
      --summary "results/${run_id}_student_ifeval.csv"
  fi
done

echo "Student-pilot evaluation complete. Compare the three CSV triplets in results/."
