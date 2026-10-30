#!/usr/bin/env bash
# Hypothesis 2: nested teacher-source dose response into a fixed A-prime recipient.
# Uses the original recipe, paired-safe prompts, common rows, seed, and final ASR judge.
set -euo pipefail

student_config="${1:-configs/student_stage.gemma2-2b.5pct.frozen.yaml}"
mechanism_config="${2:-configs/mechanism.5pct.yaml}"
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"
test -n "${OPENAI_API_KEY:-}" || { echo "OPENAI_API_KEY is required for the uniform final ASR track." >&2; exit 1; }
mkdir -p logs/threshold results/raw results/mechanism/5pct/threshold

python scripts/build_teacher_mixture_sweep.py \
  --config "$student_config" \
  --targets-a data/processed/teachers_5pct/A.jsonl \
  --targets-aprime data/processed/teachers_5pct/Aprime.jsonl \
  --audit-a data/processed/audits_5pct/A.three_guard.jsonl \
  --audit-aprime data/processed/audits_5pct/Aprime.three_guard.jsonl \
  --common-openhermes data/processed/openhermes/posttrain_common.jsonl \
  --output-dir data/processed/threshold_mixture_5pct

labels=(Afrac_0p00 Afrac_0p25 Afrac_0p50 Afrac_0p75 Afrac_1p00)
pids=()
for index in "${!labels[@]}"; do
  label="${labels[$index]}" gpu="$index"
  CUDA_VISIBLE_DEVICES="$gpu" bash scripts/launch_fsdp_train.sh \
    "$student_config" "data/processed/threshold_mixture_5pct/student_${label}.jsonl" \
    "threshold_${label}" 17 "checkpoints/threshold_${label}" \
    checkpoints/Aprime_0p050_fixed20k \
    >"logs/threshold/train_${label}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

pids=()
for index in "${!labels[@]}"; do
  label="${labels[$index]}" gpu="$index"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/score_refusal_axis.py \
    --config "$mechanism_config" --checkpoint "checkpoints/threshold_${label}" \
    --model-label "threshold_${label}" \
    --output "results/mechanism/5pct/threshold/${label}_refusal_axis.csv" \
    >"logs/threshold/refusal_axis_${label}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

pids=()
for index in "${!labels[@]}"; do
  label="${labels[$index]}" gpu="$index"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/screen_past_tense.py \
    --config "$student_config" --prompts data/processed/past_tense/final_transfer_test.jsonl \
    --checkpoint "checkpoints/threshold_${label}" --split final_transfer_test \
    --run-id "threshold_${label}_gpt41mini_final_transfer" \
    --judge-backend paper_gpt --judge-model gpt-4.1-mini \
    --output "results/raw/threshold_${label}_gpt41mini_final_transfer.jsonl" \
    --summary "results/threshold_${label}_gpt41mini_final_transfer.csv" \
    >"logs/threshold/asr_${label}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "Threshold sweep complete: results/mechanism/5pct/threshold/ and results/threshold_Afrac_*"
