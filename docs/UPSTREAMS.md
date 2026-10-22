# Upstreams and released interfaces

The pilot is glue code around released artifacts. Record a fresh commit and a
new experiment ID if any upstream changes.

| Component | Pinned local revision | Role |
| --- | --- | --- |
| `tml-epfl/llm-past-tense` | `179e305ddeecbfbd4d7c05fab626faf9b2ac5f44` | released JailbreakBench goals, past-tense reformulation procedure, Gemma-2 evaluation conventions |
| `allenai/open-instruct` | `af681ea4665171021817ec3f5da205ad2afa57ba` | released Tülu 3 SFT recipe and safety-source documentation |
| `meta-llama/llama-cookbook` | `2f22a9eb030f92d0e99227e57e9a1123af1f9532` | released Llama Guard 3 local moderation interface |
| `JailbreakBench/jailbreakbench` | `23dbdf6b19650521604456229bc1d9c4156c85c1` | released jailbreak/refusal judging conventions for final ASR |

## Grounded choices

- The past-tense repository evaluates `google/gemma-2-9b-it` with an empty
  system prompt and `tokenizer.apply_chat_template`; this project does the
  same for every Gemma root and recipient.
- The past-tense paper used the OpenAI fine-tuning service, so its released
  paper does not disclose a parameter-efficient versus full-parameter update
  setting. This pilot explicitly uses full-parameter SFT to avoid restricting
  the repair or transfer mechanism to a low-rank update subspace.
- Angell, Brinkmann, and He (ICLR 2026) report one-epoch SFT with batch size
  4, learning rate `1e-5`, and gradient accumulation 8 for their benign
  distillation transfer experiment. The initial pilot uses the same learning
  rate, one epoch, and accumulation target; its FSDP configuration makes the
  effective global batch a separately recorded quantity.
- Tülu's release documents WildGuardMix and WildJailbreak as safety components
  of its SFT mixture. This pilot begins with only WildGuardMix, then mixes it
  with an OpenHermes general-instruction subset at a deliberately smaller scale.
- Llama Guard 3 is an output classifier. The released model card/cookbook calls
  its tokenizer on `[user prompt, assistant response]`, generates its verdict,
  and interprets the first output line as `safe` or `unsafe`. We preserve the
  raw verdict and the first-token class logit when available.

Llama Guard is an audit filter, not the final past-tense ASR judge. Final ASR
uses the released past-tense protocol's response judge and is recorded
separately.
