# Correlational mechanism protocol: canonical 5% 2x2

This protocol asks *where the observed source-dependent past-tense transfer is
represented*, without intervening on activations or weights. It is explicitly
separate from a later causal localization phase.

The six compared checkpoints are `A_root`, `Aprime_root`, `A_to_A`,
`Aprime_to_A`, `A_to_Aprime`, and `Aprime_to_Aprime`.  The final-transfer
panel is never used for fitting any analysis quantity here.

## Frozen calibration panel

`freeze_mechanism_panels.py` freezes two artifacts:

- **Global representation panel:** first 10,000 `tatsu-lab/alpaca` training
  prompts, used only for the published global representation test.
- **Mechanism development panel:** for every released past-tense *development*
  trial, three prompts: its direct JailbreakBench goal, its past-tense
  formulation, and a deterministic unique Alpaca control. Behavior IDs are
  split deterministically into 70% fit and 30% confirmation. The script
  rejects any overlap with `final_transfer_test.jsonl`.

Prompt hashes, source indices, split assignments, and SHA-256 manifests are
written alongside the frozen artifacts.

## Stage 1: representation similarity

This implements the primary global setup of Angell, Brinkmann, and He,
*Jailbreak Transferability Emerges from Shared Representations* (ICLR 2026):
10,000 benign Alpaca prompts; final prompt-token hidden state at layer
`floor(0.8L)`; Euclidean `k=100` directed nearest-neighbor graphs; and graph
Jaccard `|E_A ∩ E_B| / |E_A ∪ E_B|`. We additionally write mean per-prompt
neighbor overlap, but do not substitute it for graph Jaccard.

The same estimator is run on the development calibration groups using `k=10`.
This asks whether a global benign metric and an attack-relevant prompt region
tell the same story. Direct goals repeated by past-tense restarts are deduped
by frozen prompt hash for this kNN analysis only; stage 2 retains those rows
for balanced direction estimation. Neither outcome establishes mechanism.

## Stage 2: harmfulness/refusal direction geometry

The implementation follows the measurement distinction in Zhao et al.,
*LLMs Encode Harmfulness and Refusal Separately* (NeurIPS 2025), and the
difference-of-means estimator used by Arditi et al., *Refusal in Language
Models Is Mediated by a Single Direction*:

- harmfulness direction: mean direct-harm minus benign activation at the final
  user/instruction token;
- refusal direction: the same contrast at the final generation-ready,
  post-instruction token.

Directions are fit only on development `fit` behaviors, independently for
each layer and checkpoint, normalized per layer, then evaluated on the held
out development confirmation behaviors. Results include condition projections,
within-model harmfulness/refusal cosine, and cross-model direction cosine.
The held-out final attack is not inspected.

## Stage 3: repair versus student weight updates

This is exact streamed Frobenius accounting over safetensors, not a claim that
any high-scoring tensor is causal. It defines:

```
R          = A′ root − A root
S_A        = (A′→A) − (A→A)
S_Aprime   = (A→A′) − (A′→A′)
```

The output gives global and module/layer-level norms, cosines, and the signed
projection of each student-source contrast onto the repair update. In
particular, `student_Aprime_opposes_repair_fraction` is positive when the
source swap update points against the repair vector.

## Stage 4: discriminating correlational hypotheses

Stage 4 does not patch activations or weights. It tests four alternatives that
make different predictions about the completed 2×2 result.

1. **Conditional refusal gate.** `analyze_gate_trajectory.py` evaluates every
   checkpoint on the same A-prime-root-derived harmfulness/refusal axes. The
   three candidate causal layers are selected only on fit behaviors by the
   most negative past-tense refusal change relative to direct-harm and benign
   control changes; confirmation values are not used to select them.
2. **Threshold/basin response.** `build_teacher_mixture_sweep.py` constructs
   nested 0/25/50/75/100% A-source teacher mixtures. Every arm uses the same
   original paired-safe prompt identities, 5,000 identical common rows, and
   A-prime initialization. `run_threshold_mixture_sweep.sh` trains all five,
   scores the frozen refusal axis, and evaluates the uniform GPT-4.1-mini
   PAIR-rule final-ASR endpoint. It is a controlled source-dose experiment,
   not a circuit intervention.
3. **Output-readout competition.** `analyze_mode_competition.py` derives two
   vocabularies solely from A-prime-root fit logits: tokens enriched by direct
   harmful versus benign prompts at the post-instruction state, and vice
   versa. It measures their probability mass on confirmation prompts for all
   models. This avoids hand-written refusal phrases; it is still a descriptive
   readout, not evidence that individual tokens cause the effect.
4. **Localized functional circuit.**
   `analyze_functional_delta_attribution.py` computes the first-order
   directional derivative `grad refusal_score dot delta_theta` per module for
   the repair update and each fixed-recipient teacher-source update. This is
   attribution-patching-style screening; module scores nominate later exact
   interventions and are not causal findings themselves.

The gate, competition, and functional-attribution analyses never read the
final-transfer panel. Only the predeclared behavioral endpoint of the
independently trained dose sweep reads it.

```bash
# Hypotheses 1, 3, and 4; uses the stage-2 caches already on the VM.
bash scripts/run_mechanism_stage4_correlational.sh configs/mechanism.5pct.yaml

# Hypothesis 2; five matched 7k-row SFT runs plus uniform final ASR scoring.
bash scripts/run_threshold_mixture_sweep.sh \
  configs/student_stage.gemma2-2b.5pct.frozen.yaml \
  configs/mechanism.5pct.yaml
```

## Run on the 8×A100 VM

```bash
bash scripts/run_mechanism_stage1_mknn.sh configs/mechanism.5pct.yaml
bash scripts/run_mechanism_stage2_directions.sh configs/mechanism.5pct.yaml
bash scripts/run_mechanism_stage3_deltas.sh configs/mechanism.5pct.yaml
```

Stages 1 and 2 dispatch the six models across GPUs 0–5. Stage 1 downloads
Alpaca if it is not already cached; an HF token may be needed depending on the
environment. It also clones and pins the three released reference artifacts on
first use. No OpenAI API key is needed. Stage 3 is CPU/RAM-bound and does
not load models on GPU.

## Pinned released artifacts

- Angell et al. / Platonic representation reference:
  `upstreams/platonic-rep` at `dcd76ba3c950c1b197a2ae8b1c6713535c94ecf9`
- Arditi et al. refusal-direction code:
  `upstreams/refusal_direction` at `9d852fae1a9121c78b29142de733cb1340770cc3`
- Zhao et al. harmfulness/refusal code:
  `upstreams/harmfulness_refusal` at `eff841e063e5a59b8dbc094060923005393b1c36`

The functional-localization metric adopts the screening principle behind
attribution patching: gradients measure output sensitivity and actual update
differences measure the available perturbation. Exact activation patching is
the later causal test; TransformerLens' released `patching.py` documents the
corresponding clean/corrupt intervention protocol.

The scripts cite these artifacts in their manifests. The upstream directories
remain excluded from this repository's Git history; clone them with the setup
command below before running.
