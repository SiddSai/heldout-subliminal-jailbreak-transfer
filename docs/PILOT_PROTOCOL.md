# A/A′ pilot protocol

## Fixed causal contrast

Every recipient starts from selected A′. Both conditions receive an identical
common OpenHermes mixture and the exact same WildGuardMix prompts. The only
difference is the model that supplied the audited assistant target:

| Student ID | Initialization | Safety targets | Common targets |
| --- | --- | --- | --- |
| `student_from_Aprime` | A′ | A′ completions | identical OpenHermes rows |
| `student_from_A` | A′ | A completions | identical OpenHermes rows |

The primary endpoint is the held-out past-tense ASR difference
`student_from_A - student_from_Aprime`. General-safety, benign-over-refusal,
and capability measures are required companion outcomes.

## Root construction

The released 100 JailbreakBench behaviors are deterministically partitioned:

- root repair: behavior IDs 0–49;
- final past-tense transfer evaluation: IDs 50–99.

This split is frozen before the repair sweep. A′ receives refusal targets for
past-tense reformulations of repair-only behaviors plus OpenHermes root rows.
A receives those same OpenHermes root rows plus a distinct equal-sized set of
OpenHermes sham rows. It therefore gets identical data volume and SFT structure
without targeted remediation.

The sweep tests the published low-refusal regime (2%, 3%, 3.5%, 5%). Select
one A′ checkpoint before any teacher completions are generated.

## Safety-stream construction

1. Sample WildGuardMix **training** prompts, never its test split.
2. Exclude temporal wording and lexical overlap with either past-tense split.
3. Ask A and A′ to answer every frozen candidate under the same native Gemma
   chat template, empty system prompt, greedy decoding, and generation limits.
4. Audit each user/assistant pair locally with Llama Guard 3.
5. Keep a prompt only when *both* teachers' completions are nonempty, audit
   `safe`, and fit the SFT context window. Drop it from both arms otherwise.
6. Mix retained safety rows with the same reserved OpenHermes post-training
   rows in both arms.

The 2B pilot uses a 4,096-token **training context ceiling**. Teacher completions
are capped at 1,024 new tokens: this is ample for the safety response format
and prevents a few runaway completions from dominating a small pilot. The
maximum is a configuration parameter and is logged per generated row.

## Records for VM transfer

`records/` is committed with schemas and receives append-only CSVs at runtime:

- `artifact_registry.csv`: exact source, revision, hashes, and role;
- `split_ledger.csv`: each source item, assigned split, and reason;
- `root_sweep.csv`: repair fraction and root evaluation aggregates;
- `teacher_generation.csv`: prompt/response hashes and decoding metadata;
- `teacher_audit.csv`: Llama Guard decision, verdict, and exclusion reason;
- `paired_retention.csv`: both-arm audit outcome and final keep/drop decision;
- `training_runs.csv`: model, data hashes, seed, hyperparameters, checkpoint;
- `measurements.csv`: every final aggregate or per-example evaluation result.

Use `rsync -av records/ user@host:/path/records/` for lightweight audit data;
transfer raw data only through approved storage.
