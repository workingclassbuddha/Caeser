# CAESAR: no-learning controller check (locked 2026-10-25)

## Critical falsification question

The agent observes its current velocity and the **direction of the goal**.
How much of the apparent navigation, level progression and cross-constraint
transfer can already be explained by a simple control law, with **zero learned
weights and zero optimizer steps**?

## Fixed comparison policies

All three operate solely on CAESAR's existing 14-dimensional observation:

1. `direct`: thrust toward the observed goal unit vector
   (`action = observed_goal_direction`).
2. `velocity_compensated`: thrust toward the goal while subtracting observed
   velocity (`action = clip(direction - velocity, -1, 1)`).
3. `stationary`: zero thrust, a negative control for accidental success
   and favourable spawn/goal geometry.

Controller gains and formulas above are frozen *before* test results.
No access to private goal coordinates, environment random state, simulator
internals or learned weights. All three use the **same action bounds**, time
and battery rules as CAESAR's trained actor.

The task's reward and observation expose target direction and remaining
energy explicitly; a strong hand-written controller might plausibly
solve navigation without reinforcement learning. Such a result would be
scientifically important, not an unwanted outcome.

## Paired held-out evaluation

For each of training seeds 11,23,47 and each of the eight test conditions:

- Initialize the same simulator configuration as
  `constraint_transfer.evaluate`.
- Use the exact same withheld initial state generator
  `np.random.default_rng(100000 + seed*10000 + trial)` and **explicit
  post-construction reset**.
- Execute the controller without changing any parameters or performing
  gradients.
- Log individual successes, success rate, elapsed steps, end battery and
  episode return. Keep the per-episode raw records, not just averages.
- Report each no-learning baseline's success on all test conditions and
  for the seven cross-condition test settings corresponding to each of the
  eight learned training arms.

The first **direct numerical comparison** to [PR #7's three-seed
diagnostic](https://github.com/workingclassbuddha/Caeser/actions/runs/40070005956)
uses *five* matched test trials per seed and condition. Separately assess
30 trials per seed to describe controller behaviour with more test
start/goal positions (not a direct comparison to the trained run with
only five). It is invalid to compare a five-trial learned outcome to
a 30-trial controller outcome as if both used identical samples.

## Run

```bash
python -m caesar.baselines --seeds 11 23 47 --trials 5 \
    --output outputs/controller_baselines_matched5.json
python -m caesar.baselines --seeds 11 23 47 --trials 30 \
    --output outputs/controller_baselines_30.json
```

## Decision

- If a no-learning controller matches or beats learned CAESAR at held-out
  goal success and efficiency, current task performance **does not
  establish that its observed navigation required learning**.
- If CAESAR beats no-learning controls, use more training seeds, longer
  training, matched held-out trials and an environment that defeats
  immediate goal pointing before claiming adaptive intelligence.
- Neither result proves or disproves that learning under constraints can
  yield more general intelligence; it establishes the explanatory
  strength of a simpler baseline for this particular arena.
- The archived long-run level counts are not comparable to fixed-policy
  tests without giving controls the same curriculum.
