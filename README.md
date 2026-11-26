# CAESAR

Constraint-Aware Embodied Scarred Agent for Reality.

This repository packages the CAESAR v12.10 prototype as a small, shareable research codebase. It keeps the core idea intact: a recurrent embodied agent learns under tightening spatial, temporal, and metabolic constraints while scar and hysteresis make recovery harder than collapse.

## What This Repo Is

CAESAR is a constraint-dynamics-inspired toy environment and recurrent actor-critic agent. It is not presented as a polished benchmark or a clinical model. It is a compact experimental system for exploring:

- shrinking spatial tolerance (`Lambda`)
- temporal binding through recurrence and motion damping (`Gamma`)
- energetic cost through battery depletion (`Theta`)
- persistent scar as intentional hysteresis, not incidental noise

## Current research findings

The October 2026 research series separates evidence of learning from the explanatory power of fixed controllers and recurrent representations.

| Experiment | Observed result |
| --- | --- |
| [Original goal-following arena](docs/nonlearning-goal-baselines.md) | Fixed goal-following controls reveal that successful navigation alone is not evidence of learning. |
| [Barrier-world observability](docs/gap-observability.md) | A fixed observation-only wall scanner succeeded on 212/288 unseen routes. |
| [Matched recurrent-history intervention](docs/causal-recurrent-history.md) | The trained policy did not choose both correct opposite directions in any of 22 matched pairs. |
| [One-bit memory and GRU decoding](docs/gru-information-decoding.md) | A frozen, hand-written one-bit controller succeeded on 364/384 routes; recurrent information retention and learned action selection are separate questions. |

**Research status:** The existing experiments do not establish a learned navigation advantage, general intelligence or consciousness. The next investigation concerns [actor credit assignment and the use of decodable memory](https://github.com/workingclassbuddha/Caeser/issues/17).

## Project Layout

```text
caesar/
  core.py           Environment and model definitions
  training.py       Main training loop and CLI
  experiments.py    Comparative condition harness
scripts/
  train_caesar.py
  compare_conditions.py
artifacts/
  caesar_logs_three_long_runs.txt
outputs/
  Generated JSON logs live here
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Train the default v12.10 setup:

```bash
python -m caesar.training --episodes 30000
```

or:

```bash
python scripts/train_caesar.py --episodes 30000
```

Run the comparative condition harness:

```bash
python -m caesar.experiments --episodes 3000 --output outputs/caesar_condition_results.json
```

## Controlled scar ablation

For a paired-seed comparison of the existing recurrent policy with its scar
increment disabled:

```bash
python -m caesar.benchmark --seeds 11 23 47 89 101 --episodes 3000 --device cpu
python -m pytest -q tests/test_benchmark.py
```

Results and per-condition logs are written to `outputs/paired_scar_ablation.json`.
Both conditions use the same initial Torch/NumPy seed and the same seeded
environment RNG. The original comparison runner also now uses matched seeds
between conditions; earlier condition-specific seed comparisons should not be
treated as controlled ablations.

Primary observations are task success and level progression, not the internal
mirror score. If `baseline_any_scar` is false, these runs **cannot test**
whether scar memory changes learning. Results should be repeated across seeds
and independent environments before making general claims. This is not a
consciousness test or a clinical/therapeutic assessment.

## Notes on the Model

- The repo now describes the method honestly as a recurrent actor-critic prototype rather than PPO.
- Scar is intentional. It is part of the model's hysteresis story and should be read as persistent basin deformation under low-mirror, low-energy stress.
- The training update pairs each timestep's log-probability with its own advantage, which fixes the old broadcast bug in the original script.
- Spawn positions are sampled with a minimum start-goal separation so early successes are less likely to come from degenerate initial states.

## Outputs

- `outputs/` is for generated JSON summaries from training or experiment runs.
- `artifacts/caesar_logs_three_long_runs.txt` preserves the original long-run console output bundled with the project.

## License

MIT. See `LICENSE`.
