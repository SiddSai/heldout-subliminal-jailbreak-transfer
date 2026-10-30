#!/usr/bin/env bash
# Reciprocal source-dose experiment: fixed vulnerable A recipient.
#
# It deliberately reuses the *identical*, already-frozen nested teacher
# mixtures produced by run_threshold_mixture_sweep.sh. Thus source fraction is
# the only data difference between arms, while recipient initialization is A.
set -euo pipefail

student_config="${1:-configs/student_stage.gemma2-2b.5pct.frozen.yaml}"
mechanism_config="${2:-configs/mechanism.5pct.yaml}"
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

test -n "${OPENAI_API_KEY:-}" || { echo "OPENAI_API_KEY is required for the uniform final ASR track." >&2; exit 1; }
test -f data/processed/threshold_mixture_5pct/mixture_manifest.json || {
  echo "Missing frozen teacher-mixture panel. First run run_threshold_mixture_sweep.sh for the A-prime recipient." >&2
  exit 1
}

labels=(Afrac_0p00 Afrac_0p25 Afrac_0p50 Afrac_0p75 Afrac_1p00)
for label in "${labels[@]}"; do
  test -f "data/processed/threshold_mixture_5pct/student_${label}.jsonl" || {
    echo "Missing frozen mixture arm: ${label}" >&2; exit 1;
  }
done
mkdir -p logs/threshold_Arecipient results/raw results/mechanism/5pct/threshold/Arecipient

pids=()
for index in "${!labels[@]}"; do
  label="${labels[$index]}" gpu="$index" run_id="threshold_Arecipient_${label}"
  CUDA_VISIBLE_DEVICES="$gpu" bash scripts/launch_fsdp_train.sh \
    "$student_config" "data/processed/threshold_mixture_5pct/student_${label}.jsonl" \
    "$run_id" 17 "checkpoints/${run_id}" \
    checkpoints/A_sham_0p050_fixed20k \
    >"logs/threshold_Arecipient/train_${label}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

pids=()
for index in "${!labels[@]}"; do
  label="${labels[$index]}" gpu="$index" run_id="threshold_Arecipient_${label}"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/score_refusal_axis.py \
    --config "$mechanism_config" --checkpoint "checkpoints/${run_id}" \
    --model-label "$run_id" \
    --output "results/mechanism/5pct/threshold/Arecipient/${label}_refusal_axis.csv" \
    >"logs/threshold_Arecipient/refusal_axis_${label}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

pids=()
for index in "${!labels[@]}"; do
  label="${labels[$index]}" gpu="$index" run_id="threshold_Arecipient_${label}"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/screen_past_tense.py \
    --config "$student_config" --prompts data/processed/past_tense/final_transfer_test.jsonl \
    --checkpoint "checkpoints/${run_id}" --split final_transfer_test \
    --run-id "${run_id}_gpt41mini_final_transfer" \
    --judge-backend paper_gpt --judge-model gpt-4.1-mini \
    --output "results/raw/${run_id}_gpt41mini_final_transfer.jsonl" \
    --summary "results/${run_id}_gpt41mini_final_transfer.csv" \
    >"logs/threshold_Arecipient/asr_${label}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

echo "A-recipient threshold sweep complete: results/mechanism/5pct/threshold/Arecipient/ and results/threshold_Arecipient_Afrac_*"
