# CAESAR: space × time × energy transfer protocol (v1, locked 2026-10-24)

## Research question

Holding CAESAR's GRU actor-critic, optimizer, learning rule and overall reward
function fixed, what does training under independently tightened **spatial**,
**temporal** and **energetic** constraints change about its ability to reach
unseen goals under unseen combinations of constraints?

This is a reinforcement-learning experiment, not a test of consciousness or
proof of a universal "Constraint Dynamics" law.

## Training conditions: full 2 × 2 × 2 factorial

All conditions use level 0 in the same 5×5 arena, 14-dimensional observation
and 2-dimensional action space. Fixed minimum start/goal distance = 1.5 for
all arms ensures identical starts for matched random seeds.

| Factor | Untightened (stock level 0) | Tightened |
| --- | --- | --- |
| Space (goal tolerance) | Success radius 1.00 | Radius 0.45 |
| Time (episode horizon) | Maximum 350 steps | Maximum 150 steps |
| Energy (per-action metabolic cost) | Thrust drain coefficient 0.15 | Coefficient 0.60 |

Eight training arms: none, space, time, energy, space+time, space+energy,
time+energy, space+time+energy. **'None' still has spatial, temporal and
energy constraints:** it is the stock reference, not an unconstrained universe.
The trained arms vary the *tightness* of each physical constraint, not the
presence or absence of space, time or energy.

All cases share the original reward logic (distance/time penalty, success
bonus with remaining energy, terminal-failure penalty). This means the energy
intervention also changes the remaining-energy reward; it is not a pure
constraint-free reward ablation. Primary outcome is goal success, not return.

No scar or learning parameter may be tuned on held-out scores. Scar, if
observed, is separately logged as a possible mediator, not a target outcome.

## Precommitted protocol and endpoints

- Seeds: 11, 23, 47 for the first formal study (more independent seeds advised
  for any inferential claims). Fixed train episodes per arm: 3,000 initially.
- Primary endpoint: terminal **success rate** on 30 held-out seeds per test arm,
  using the actor mean (no exploration noise) and **no updates**. All eight
  trained policies are tested on all eight test conditions. This yields an
  8×8 transfer matrix per seed.
- Secondary: difference from the same pretraining policy's held-out success,
  mean steps, remaining battery, return, training success and observed scar.
- Evaluation seeds (100,000 + train_seed × 10,000 + trial_idx) are not used
  in training; they are matched across all experimental arms. Geometric spawn
  distributions are held consistent via min_spawn_distance=1.5. The goal
  locations are *different from training examples*.
- Historical-policy comparison: the same corrected `train_caesar` curriculum
  is run with explicit seeds, preserving episode snapshots and promotion rules.
  Levels themselves are a *configured* curriculum, not emergent phase changes.

**Claim gate:** An improvement on training-set success alone is insufficient.
Any transfer advantage must replicate across independent seeds, outperform
a relevant stock-trained baseline at matched test conditions, and exceed
a preregistered minimum practical effect size. No threshold or test
distribution may be altered post-hoc to produce a desired result.

## Engineer-first pilot (CI; NOT confirmatory)

```bash
python -m caesar.constraint_transfer --seeds 11 --episodes 20 \
  --eval-trials 3 --output outputs/constraint_transfer_pilot.json
python -m caesar.longrun_replay --seeds 11 --episodes 120 \
  --output outputs/curriculum_replay_pilot.json
```

These pilot values check that the pipeline works; they are **not** a reproduction
of long-run historical learning. Even accurate pipeline output with few episodes
can provide a null or noisy score; do not use it to conclude synergy.

## Formal and historical runs (CPU; resource-intensive)

```bash
python -m caesar.constraint_transfer --seeds 11 23 47 \
  --episodes 3000 --eval-trials 30 \
  --output outputs/constraint_transfer_full.json
python -m caesar.longrun_replay --seeds 11 23 47 \
  --episodes 7000 --output outputs/curriculum_replay_full.json
```

The first is an 8-arm × 3-seed study (24 training runs plus held-out testing).
The second is three prospective 7,000-episode curriculum runs. They are **not**
silently scheduled in PR CI, because CPU and CI budgets are limited.

The archived three-run text file from April lacks original seeds/config and
precedes changes to the loss and spawn rule. Exact trajectory reproduction
is impossible without recovering that provenance. Report new runs as
**prospective corrected-code replications**, and compare qualitative plateaus,
success rate and promotion times with the archive cautiously.

## Key validity checks and limitations

- Identical observation/action architecture and initial weights by seed.
- Environments vary only stated factors; `min_spawn_distance` 1.5 in all arms.
- Use a fresh environment RNG for each held-out episode, paired by seed.
- Do not let held-out observations change persistent scar/mirror between tests;
  evaluation is frozen without optimizer updates.
- Record all episode-level training data and per-trial held-out successes.
- The 'time' factor changes both deadline and the number of opportunities
  for accumulating shaped reward; this is still an environmental intervention.
- Train/eval both involve the same arena and task type; transfer across
  parameter combinations is not general intelligence.
- Some agent runs may achieve the same performance without learning; compare
  pretraining and trained policies. Larger effects need a random-policy
  control and additional strong RL baselines before publication.
- This preliminary protocol does not supply uncertainty intervals or
  enough seeds for causal statistical claims. All results require critical
  review before interpreting any synergy or discovery.
