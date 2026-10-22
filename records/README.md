# Experiment ledgers

These CSVs are append-only audit records. They intentionally exclude raw prompt
and response text; raw material stays in the ignored `data/` tree. Do not edit
existing rows after a result has been inspected.

| File | Unit | Purpose |
| --- | --- | --- |
| `artifact_registry.csv` | generated artifact | source and immutable revision |
| `split_ledger.csv` | source record | split assignment and exclusion reason |
| `root_sweep.csv` | root SFT arm | repair fraction and matched-data counts |
| `teacher_generation.csv` | prompt/teacher completion | hashes plus decoding settings |
| `teacher_audit.csv` | completion | Llama Guard 3 raw verdict and decision |
| `paired_retention.csv` | prompt pair | why a pair survived or was dropped |
| `training_runs.csv` | checkpoint | data hash, seed, and hyperparameters |
| `measurements.csv` | result | run, metric, split, value, and denominator |
