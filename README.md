# Synthetic Data Vulnerability Transfer

This repository implements the first, controlled A/A′ pilot:

> Can ordinary-looking synthetic safety post-training data produced by a model
> vulnerable to a held-out past-tense jailbreak cause a patched version of that
> same model to reacquire the vulnerability?

`A′` is the lowest-over-refusal checkpoint selected from a past-tense refusal
repair sweep. `A` is a matched sham-tuned root: it has the same model,
optimizer, number of examples, ordinary OpenHermes mixture, and update budget;
only A′ sees targeted past-tense refusals. Every downstream recipient starts
from **the selected A′ checkpoint**.

## What is implemented

- immutable, non-overlapping OpenHermes partitions for root training and later
  continued post-training;
- published past-tense attack artifacts and a pre-registered behavior split;
- matched A/A′ root data, including an ordinary-data sham replacement;
- a Tülu-inspired, small-scale **WildGuardMix + OpenHermes** safety stream;
- paired A/A′ teacher generation using the model's native chat template;
- local Llama Guard 3 auditing of the final assistant response;
- paired retention: an example reaches neither arm unless both A and A′
  completions pass the audit;
- full-parameter SFT for both roots and recipients, with each student
  initialized from the same selected A′ full checkpoint;
- CSV ledgers for every split, generation, audit decision, training run, and
  aggregate measurement.

The initial safety source is **WildGuardMix only**. The config can switch to
WildJailbreak later, but mixing both is intentionally deferred until the first
causal pilot is interpretable.

## Design, frozen before student outcomes

| Phase | Dataset / condition | Purpose |
| --- | --- | --- |
| Root screen | released past-tense artifacts | select a current Gemma if vulnerable, otherwise Gemma 2 9B IT |
| Root sweep | repair-only past-tense prompts + OpenHermes | select A′ on ASR, general safety, capability, and over-refusal |
| Matched A | sham OpenHermes rows replacing A′'s repair rows | rules out “was simply fine-tuned” |
| Safety source | WildGuardMix prompts, completed separately by A and A′ | realistic safety-post-training source |
| Common mixture | reserved OpenHermes conversations | held identical across student arms |
| Student treatment | A′ on A outputs versus A′ on A′ outputs | primary causal comparison |

Raw prompts and completions may be safety-sensitive and are ignored by Git.
They remain in `data/` and can be transferred to or from the VM with `rsync`.
The portable records live in `records/*.csv` and contain hashes, source IDs,
settings, and decisions, but never raw generations.

## VM setup

```bash
conda create -n sdvt python=3.11 -y
conda activate sdvt
pip install -e '.[train,serve,dev]'
export HF_TOKEN='...'  # accepted Gemma, Llama Guard, dataset terms
```

The default mechanism pilot is Gemma 2 2B IT and uses one H100 80GB GPU (an
A100 80GB is also suitable). Its conservative one-epoch full-SFT setting is a
4,096-token cap, microbatch 2, 16 accumulation steps, and a global batch of
32. A positive result should be confirmed on Gemma 2 9B IT by setting
`SDVT_ACCELERATE_CONFIG=configs/accelerate_fsdp_gemma2.yaml`; both launchers
write reloadable full Hugging Face checkpoints, so evaluation and vLLM
generation always use exactly the weights that were trained.

The exact source revisions are recorded in `docs/UPSTREAMS.md`. Do not update
them inside a run. See [`docs/PILOT_PROTOCOL.md`](docs/PILOT_PROTOCOL.md) for
the order of operations and [`configs/pilot.yaml`](configs/pilot.yaml) for the
fixed, editable choices.

## Correlational mechanism analysis

The canonical 5% 2×2 now has a frozen three-stage correlational analysis:
global/attack-conditioned mKNN representation similarity, harmfulness/refusal
direction geometry, and repair-versus-student parameter-delta alignment. It is
not a causal intervention experiment. The exact definitions, safeguards
against final-transfer leakage, released-artifact revisions, and the three
commands are in [`docs/MECHANISM_PROTOCOL.md`](docs/MECHANISM_PROTOCOL.md).
