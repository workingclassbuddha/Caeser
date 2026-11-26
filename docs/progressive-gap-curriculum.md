# CAESAR v2: can a progressively tightened physical constraint enable learning?

## Why this experiment exists

The first v2 fixed-gap arena is **reachable** by a reference route (100%
of tested instances), but two no-learning direct-to-goal controllers score
0%, and CAESAR's unchanged recurrent learner managed **only one training
success among 450 episodes** in the 3-seed ×150-episode diagnostic.
The latter is an empirical negative result at the chosen budget, not a
universal proof of unlearnability.

Changing the neural network or explicitly giving it a route would obscure
the user's central constraint-dynamics question. This experiment varies
ONLY the physical width of the wall passage over training time.

## Independent variable

Both arms use the identical CAESAR network, training code, optimizer, reward,
battery, action/observation count, arena, start/goal distribution and seeded
layout selection.

1. **fixed_hard**: gap half-width 0.55 from episode 0 onwards.
2. **graduated**: gap half-width 2.1 at episode 0 (direct route accessible),
   continuously narrows to 0.55 by 70% of training, then stays at 0.55
   for the remaining 30% of episodes.

```python
width = 2.1 - (2.1 - 0.55) * min(1, episode / (0.70 * total_episodes))
```

No reward shaping, explicit routing waypoints, memory modules or optimizer
changes are introduced. Crucially this is **designed curriculum
learning**, which is a stronger intervention than simply exposing an
untrained agent to three abstract constraints. Success would support
a *conditional learning-from-progressive-constraints* claim, not learning
from literally no other algorithmic or environmental structure.

## Primary endpoint

Held-out success of the **frozen policy** on the ORIGINAL FULLY HARD gap
(half-width 0.55, seed-disjoint layouts) after identical training budgets.
Compare matched seeds' graduated-minus-fixed-hard outcomes and
pre/posttraining changes. Secondary outcomes are episode-level training
success, last-third hard-training performance and transfer across the
eight original v2 test constraint combinations.

Every outcome must include per-episode width and objective success and the
full per-trial held-out results, so we can tell if improvement merely
arises during easy early episodes.

## Exact commands

Engineering feasibility pilot:

```bash
python -m caesar.challenge_curriculum --seeds 11 23 \
    --episodes 90 --trials 4 \
    --output outputs/curriculum_v2_pilot.json
```

Prospective confirmatory scale *after* code and feasibility review:

```bash
python -m caesar.challenge_curriculum --seeds 11 23 47 89 101 \
    --episodes 3000 --trials 30 \
    --output outputs/curriculum_v2_full.json
```

The first is not powered to show a reliable adaptive advantage. The
second is computationally expensive and is not part of the PR smoke CI.

## Decision policy

- If graduated training eventually achieves better full-hard held-out
  success across independent seeds, compare it with observation-only
  wall-following and map-aware controllers before claiming substantive
  learned intelligence.
- If the agent only improves during wide openings or fails at full
  difficulty, the curriculum did **not** demonstrate robust adaptation.
- If both schedules fail, report that finding and inspect training
  credit assignment and observability as limitations. Do not redefine
  success or alter parameters post-hoc while treating the revised
  outcome as confirmatory.
- Archived v1 level counters and v2 feasibility-planner results are
  not evidence that a learner solved the obstacle.

The aim is to investigate the conditions in which a generic learning
system discovers strategies under constraints—not a proof of universal
cognition or consciousness.
