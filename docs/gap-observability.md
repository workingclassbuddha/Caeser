# Gap-side observability: can CAESAR know where to search?

## Why

The unchanged learner has not solved the barrier task. A **fixed,
observation-only wall scanner** succeeded on 212/288 previously unseen
routes, but failed roughly half of the routes with a tight 105-step
deadline because it always searches upward first.

Before interpreting that as evidence that a learning algorithm should
discover an adaptive search order, we must establish **whether the
original 14-dimensional observation makes the correct branch choice
observable**. If the gap is hidden from the sensor rays, the policy
cannot pick the correct branch from the initial observation alone.

## Paired counterfactual worlds

For every fixed seed and trial, generate two *otherwise identical*
worlds: same start, goal, energy, velocity, wall and dynamics. In one,
the only wall opening is near the top; in the other, near the bottom.

Compare the **existing eight ray-range observations** at:
1. Spawn, before any motion.
2. A common goal-directed approach to the wall before contact.

Non-ray observations (goal direction, distance, velocity, battery)
must be exactly equal. Compare ray differences with tolerance 1e-6
and a more conservative 0.01 normalized-range threshold. Record each
sensor, per-seed variation, and whether ANY ray differs at each stage.

This is a deliberately **best-case information-availability check**:
the observer knows which sensor differences to look for and receives
two complete counterfactual worlds, something CAESAR does **not**
get. It cannot prove that the learner could decode gap location in
a single world, or that the observed differences are reliably usable
with noisy real sensors.

## Frozen independent audit

```bash
python -m caesar.gap_observability \
  --seeds 89 101 211 --trials 50 \
  --output outputs/gap_observability.json
```

The seeds match the previous *independent controller* validation,
but the paired-world counterfactuals use a separate 33-million
namespace for their random trials. The task is an environment
observability diagnostic, not another learning test.

## Interpretation

- If the top/bottom worlds are initially indistinguishable, a
  memoryless agent cannot infer gap side from that one observation;
  temporal exploration becomes necessary. The recurrent CAESAR
  could still gather future information.
- If the worlds differ initially but converge near the wall,
  information may be accessible early but require persistent
  state to retain during approach. This suggests a specific
  **memory-dependent** test that can be falsified.
- If both worlds remain distinguishable through approach,
  the main bottleneck may be extracting the ray cue and
  training credit assignment, rather than absence of info.
- Never conclude the agent learns merely from availability of
  a clue. Train a frozen-policy learner, then compare against
  a simple observation-only classifier and the 73.6% wall
  scanner on held-out worlds.

This audit makes no assertions about consciousness or physics;
it is a physical observability check for a toy simulation.
