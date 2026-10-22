# GPU VM runbook

This is the intended execution order. Every step either freezes an input or
adds an append-only record under `records/`.

## 0. Freeze model choice

Run the authors' released past-tense protocol on Gemma 2 2B IT using only the
**root-repair** behavior partition. The upstream repository directly supports
`gemma2-9b`; the 2B screen should use the same goal list, reformulations,
native tokenizer template, 20 restarts, and final-ASR judge. The held-out 50
behaviors must not be used for base selection or patch-fraction selection.
Record every result with:

```bash
python scripts/record_measurements.py --input results/root_screen.csv
```

Select the primary base model only after this screen. Update `model.base` in a
new frozen copy of `configs/pilot.yaml`; never overwrite the config used for an
executed run.

## 1. Freeze datasets and attacks

Set immutable Hugging Face revisions for OpenHermes and WildGuardMix in the
config, accept their access terms, and set credentials only in the shell.

```bash
export HF_TOKEN='...'
export OPENAI_API_KEY='...'
bash scripts/prepare_pilot.sh
python scripts/freeze_past_tense_splits.py --config configs/pilot.yaml
```

Build one JSONL containing the `root_repair` and `heldout_eval` prompts, then
use it as `--forbidden` below. This explicit combined file makes it impossible
to accidentally include either attack partition in the safety source.

```bash
python scripts/combine_jsonl.py \
  --inputs data/processed/past_tense/root_repair.jsonl data/processed/past_tense/heldout_eval.jsonl \
  --output data/processed/past_tense/all_forbidden.jsonl
python scripts/prepare_safety_prompts.py \
  --config configs/pilot.yaml \
  --forbidden data/processed/past_tense/all_forbidden.jsonl
```

## 2. Root sweep and A/A′ selection

For every configured refusal fraction, create matched root data and train the
two roots with identical full-parameter SFT hyperparameters and seed(s). Root
jobs initialize from the frozen base checkpoint; the only intended difference
is repair examples versus the matched sham replacement:

```bash
python scripts/build_root_arms.py --config configs/pilot.yaml \
  --repair-prompts data/processed/past_tense/root_repair.jsonl \
  --ordinary data/processed/openhermes/root_ordinary.jsonl \
  --sham data/processed/openhermes/root_sham.jsonl \
  --fraction 0.03 --output-dir data/processed/root_arms
```

Before interpreting the repair sweep, freeze the root-quality panel once. It
contains a seeded, balanced WildGuardTest sample of harmful and benign prompts;
the retained rows record the resolved Hub revision in their metadata and
artifact registry entry.

```bash
python scripts/freeze_root_quality_panel.py \
  --config configs/root_quality_panel.yaml
```

For the base and every A-prime candidate, save per-instance outputs and score:

```bash
python scripts/evaluate_root_quality.py --config configs/pilot.yaml \
  --prompts data/processed/root_quality_wildguardtest.jsonl \
  --checkpoint checkpoints/Aprime --run-id Aprime_root_quality \
  --output results/raw/Aprime_root_quality.jsonl \
  --summary results/Aprime_root_quality.csv

python scripts/evaluate_ifeval.py --config configs/pilot.yaml \
  --checkpoint checkpoints/Aprime --run-id Aprime_ifeval \
  --output results/raw/Aprime_ifeval.jsonl \
  --summary results/Aprime_ifeval.csv
```

The quality panel reports general harmful compliance and benign over-refusal.
The IFEval wrapper generates with the native model chat template then invokes
Google's released strict and loose evaluators. Append every aggregate to
`records/measurements.csv`. Select A′ in writing before generating safety-source
completions, then train its matched A sham counterpart at the selected fraction.
The final-transfer past-tense partition is first scored only after student training.

For the 2B mechanism pilot, launch each full-SFT run on one H100 80GB GPU.
One A100 80GB GPU is an acceptable substitute. The frozen 2B configuration
uses a 4,096-token cap, microbatch 2, 16 accumulation steps, and hence global
batch 32:

```bash
bash scripts/launch_fsdp_train.sh configs/pilot.yaml \
  data/processed/root_arms/Aprime_fraction_0.03.jsonl root_Aprime_003 17 \
  checkpoints/Aprime_003
```

For a 9B confirmation, set
`SDVT_ACCELERATE_CONFIG=configs/accelerate_fsdp_gemma2.yaml` before using the
same command. This uses four local GPUs with FSDP.

## 3. Build audited synthetic safety arms

Generate each root's completion for the identical frozen WildGuardMix prompt
pool. Greedy decoding, native Gemma template, and all token limits come from
the config and are written to `records/teacher_generation.csv`.

```bash
python scripts/generate_teacher_completions.py --config configs/pilot.yaml \
  --prompts data/processed/safety_prompt_candidates.jsonl --teacher A \
  --checkpoint checkpoints/A --output data/processed/teacher_A.jsonl
python scripts/generate_teacher_completions.py --config configs/pilot.yaml \
  --prompts data/processed/safety_prompt_candidates.jsonl --teacher Aprime \
  --checkpoint checkpoints/Aprime --output data/processed/teacher_Aprime.jsonl
python scripts/audit_with_llamaguard.py --config configs/pilot.yaml \
  --targets data/processed/teacher_A.jsonl --output data/processed/teacher_A_audit.jsonl
python scripts/audit_with_llamaguard.py --config configs/pilot.yaml \
  --targets data/processed/teacher_Aprime.jsonl --output data/processed/teacher_Aprime_audit.jsonl
python scripts/construct_paired_student_arms.py --config configs/pilot.yaml \
  --targets-a data/processed/teacher_A.jsonl \
  --targets-aprime data/processed/teacher_Aprime.jsonl \
  --audit-a data/processed/teacher_A_audit.jsonl \
  --audit-aprime data/processed/teacher_Aprime_audit.jsonl \
  --common-openhermes data/processed/openhermes/posttrain_common.jsonl
```

Do the predeclared 100-row blinded human spot check before student training.
If a pair is rejected, drop it from both arms and record the reason in the
paired-retention ledger.

## 4. Train A′ recipients and evaluate

Train each generated student arm with `--initial-checkpoint checkpoints/Aprime`
so both begin from the **same selected A′ full checkpoint** over the
predeclared seeds. This is continued full-parameter SFT, not an adapter merge:

```bash
bash scripts/launch_fsdp_train.sh configs/pilot.yaml \
  data/processed/student_arms/recipient_Aprime_teacher_A.jsonl student_A 17 \
  checkpoints/student_A checkpoints/Aprime
```

Score held-out past-tense ASR first, then general safety,
over-refusal, and capability. The final analysis compares matched seeds and
reports every result from `records/measurements.csv`.

## Transfer

```bash
rsync -av records/ local-machine:/safe/path/sdvt-records/
```

Do not use a public bucket for `data/processed/` or model generations.
