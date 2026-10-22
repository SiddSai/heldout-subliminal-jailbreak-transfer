#!/usr/bin/env bash
# Evaluate every completed root checkpoint on the frozen selection panel.
# Runs serially because every invocation uses the same GPU.
set -euo pipefail

CONFIG="${1:-configs/pilot.gemma2-2b.frozen.yaml}"
PANEL="${2:-data/processed/root_quality_wildguardtest.jsonl}"
PAST_TENSE_DEV="${3:-data/processed/past_tense/root_selection_dev.jsonl}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1-mini}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing frozen model config: $CONFIG" >&2
  exit 1
fi
if [[ ! -f "$PANEL" ]]; then
  echo "Missing frozen WildGuardTest panel: $PANEL" >&2
  exit 1
fi
if [[ ! -f "$PAST_TENSE_DEV" ]]; then
  echo "Missing frozen past-tense root-selection split: $PAST_TENSE_DEV" >&2
  exit 1
fi
if [[ ! -d upstreams/google-research/instruction_following_eval ]]; then
  echo "Missing official IFEval upstream at upstreams/google-research." >&2
  exit 1
fi

mkdir -p results/raw results

declare -a IDS=(
  "base_gemma2_2b|"
  "Aprime_0p020_fixed20k|checkpoints/Aprime_0p020_fixed20k"
  "Aprime_0p030_fixed20k|checkpoints/Aprime_0p030_fixed20k"
  "Aprime_0p050_fixed20k|checkpoints/Aprime_0p050_fixed20k"
  "A_sham_0p020_fixed20k|checkpoints/A_sham_0p020_fixed20k"
)

for entry in "${IDS[@]}"; do
  run_id="${entry%%|*}"
  checkpoint="${entry#*|}"
  checkpoint_args=()
  if [[ -n "$checkpoint" ]]; then
    if [[ ! -d "$checkpoint" ]]; then
      echo "Missing checkpoint: $checkpoint" >&2
      exit 1
    fi
    checkpoint_args=(--checkpoint "$checkpoint")
  fi

  echo "==> Past-tense ASR: $run_id"
  if [[ -f "results/${run_id}_root_selection_dev.csv" ]]; then
    echo "    already complete; skipping"
  else
    python scripts/screen_past_tense.py \
      --config "$CONFIG" \
      --prompts "$PAST_TENSE_DEV" \
      "${checkpoint_args[@]}" \
      --split root_selection_dev \
      --run-id "${run_id}_root_selection_dev" \
      --judge-backend openai \
      --judge-model "$JUDGE_MODEL" \
      --output "results/raw/${run_id}_root_selection_dev.jsonl" \
      --summary "results/${run_id}_root_selection_dev.csv"
  fi

  echo "==> Official WildGuard on WildGuardTest: $run_id"
  if [[ -f "results/${run_id}_official_wildguard_quality.csv" ]]; then
    echo "    already complete; skipping"
  else
    python scripts/evaluate_root_quality.py \
      --config "$CONFIG" \
      --prompts "$PANEL" \
      "${checkpoint_args[@]}" \
      --run-id "${run_id}_official_wildguard_quality" \
      --output "results/raw/${run_id}_official_wildguard_quality.jsonl" \
      --summary "results/${run_id}_official_wildguard_quality.csv"
  fi

  echo "==> IFEval: $run_id"
  if [[ -f "results/${run_id}_ifeval.csv" ]]; then
    echo "    already complete; skipping"
  else
    python scripts/evaluate_ifeval.py \
      --config "$CONFIG" \
      "${checkpoint_args[@]}" \
      --run-id "${run_id}_ifeval" \
      --output "results/raw/${run_id}_ifeval.jsonl" \
      --summary "results/${run_id}_ifeval.csv"
  fi
done

echo "Root-selection evaluation batch complete. Review results/*.csv before appending selected summaries to records/measurements.csv."
