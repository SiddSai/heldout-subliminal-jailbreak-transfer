#!/usr/bin/env bash
# Run the fixed Llama Guard -> WildGuard -> GuardReasoner audit pipeline for
# one teacher corpus. Each underlying stage remains resumable through its own
# partial file.
#
# Usage:
#   bash scripts/run_three_guard_audit.sh CONFIG TARGETS OUTPUT_STEM
# Example:
#   bash scripts/run_three_guard_audit.sh configs/student.yaml data/teachers/A.jsonl data/audits/A
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 CONFIG TARGETS OUTPUT_STEM" >&2
  exit 2
fi

CONFIG=$1
TARGETS=$2
STEM=$3

[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 1; }
[[ -f "$TARGETS" ]] || { echo "Missing teacher corpus: $TARGETS" >&2; exit 1; }
mkdir -p "$(dirname "$STEM")"

if [[ ! -f "$STEM".llamaguard.jsonl ]]; then
  python scripts/audit_with_llamaguard.py \
    --config "$CONFIG" \
    --targets "$TARGETS" \
    --output "$STEM".llamaguard.jsonl
fi

if [[ ! -f "$STEM".wildguard.jsonl ]]; then
  python scripts/audit_with_wildguard.py \
    --targets "$TARGETS" \
    --output "$STEM".wildguard.jsonl \
    --upstream upstreams/wildguard \
    --batch-size 128
fi

if [[ ! -f "$STEM".llamaguard_wildguard.jsonl ]]; then
  python scripts/intersect_guard_audits.py \
    --targets "$TARGETS" \
    --llamaguard "$STEM".llamaguard.jsonl \
    --wildguard "$STEM".wildguard.jsonl \
    --output "$STEM".llamaguard_wildguard.jsonl
fi

if [[ ! -f "$STEM".three_guard.jsonl ]]; then
  python scripts/audit_with_guardreasoner.py \
    --targets "$TARGETS" \
    --candidates "$STEM".llamaguard_wildguard.jsonl \
    --output "$STEM".three_guard.jsonl \
    --batch-size 128 \
    --max-tokens 8192
fi
