#!/usr/bin/env bash
# Stage 3: streamed exact repair-vs-student parameter-delta alignment.
set -euo pipefail

config="${1:-configs/mechanism.5pct.yaml}"
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"
mkdir -p logs/mechanism
python scripts/analyze_parameter_deltas.py --config "$config" \
  | tee logs/mechanism/stage3_parameter_deltas.log
echo "Stage 3 complete: results/mechanism/5pct/stage3_parameter_deltas/"
