#!/usr/bin/env bash
# Fetch the released reference artifacts at the revisions recorded in the
# frozen mechanism config. Existing checkouts must already be at that revision.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

ensure_upstream () {
  local path="$1" url="$2" revision="$3"
  if [[ -d "$path/.git" ]]; then
    local actual
    actual="$(git -C "$path" rev-parse HEAD)"
    [[ "$actual" == "$revision" ]] || {
      echo "Pinned revision mismatch for $path: have $actual, need $revision" >&2
      exit 1
    }
    return
  fi
  [[ ! -e "$path" ]] || { echo "Cannot clone: $path exists but is not a Git checkout" >&2; exit 1; }
  git clone "$url" "$path"
  git -C "$path" checkout --detach "$revision"
}

ensure_upstream upstreams/refusal_direction \
  https://github.com/andyrdt/refusal_direction.git \
  9d852fae1a9121c78b29142de733cb1340770cc3
ensure_upstream upstreams/harmfulness_refusal \
  https://github.com/CHATS-lab/Llms_Encode_Harmfulness_Refusal_Separately.git \
  eff841e063e5a59b8dbc094060923005393b1c36
ensure_upstream upstreams/platonic-rep \
  https://github.com/minyoungg/platonic-rep.git \
  dcd76ba3c950c1b197a2ae8b1c6713535c94ecf9
ensure_upstream upstreams/llm-past-tense \
  https://github.com/tml-epfl/llm-past-tense.git \
  179e305ddeecbfbd4d7c05fab626faf9b2ac5f44
[[ -f upstreams/llm-past-tense/harmful_behaviors_jailbreakbench.csv ]] || {
  echo "Past-tense checkout is missing harmful_behaviors_jailbreakbench.csv" >&2
  exit 1
}
echo "Pinned mechanism upstreams are ready."
