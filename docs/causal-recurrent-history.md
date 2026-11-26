# CAESAR v2: causal intervention on recurrent history (2026-10-25)

## Research question

Earlier v2 tests showed that an opening's position affects early diagonal
ray observations in 150/150 matched world pairs, but only 4/150 near-wall
pairs differ by a normalized range greater than 0.01. Does CAESAR **use
information from that earlier observation to choose correctly** at the wall?

A recurrent network is *capable* of different actions with different
histories even before it learns. Therefore a hidden-state difference by
itself is **not** evidence of learned memory or pathfinding.

## Strict matched-history intervention

- For each trial, make two physically identical worlds (same start, goal,
  dynamics, energy and velocity). Only hidden passage side (top vs bottom)
  differs. No model is shown the side label.
- Drive both to the same near-wall position with **the identical fixed
  goal-directed actions**, recording every 14-value observation. This
  forced approach is part of the diagnostic only; it is not learned
  navigation.
- For a single frozen model, replay either observed history through the
  GRU and calculate action from the final observation with two variants:
  1. **Preserved**: recurrent state comes from early-to-late observations.
  2. **Reset**: replace recurrent state with zeros immediately before the
     exact same final observation is fed to the network.
- Restrict the strongest causal claim to pairs with nearly identical
  final 14-vectors (all absolute differences below 1e-6). Where input
  differs, report separately rather than attributing action differences
  to memory.
- Recurrent-model parameters, mirror and scar buffers are frozen during
  the probe. Both regimes see the same final stimulus.
- Report vertical action sign for top/bottom as well as pair-action
  difference. A remembered cue counts as **correctly used** only if
  the policy chooses positive y for a top gap and negative y for a
  bottom gap, on each pair. Also report the fraction of trial pairs
  where *both* choices are correct and compare untrained versus trained
  policies at the same seeds. Mere nonzero hidden/action contrasts
  do NOT satisfy the success gate.

We test randomly initialized and actor-critic-trained networks with the
same architecture and original training rule. The short CI run uses 2
training seeds, 50 episodes per model, 12 probe pairs per seed; this
is not enough to produce reliable route learning.

## Nonlearning one-bit memory control

At the FIRST observation, compare the northeast and southeast diagonal
ray ranges (indices 7 and 13 in the 14-observation array). Store exactly
one bit: preferred opening side. When approaching the wall, a finite
state scanner uses that stored bit to scan in the preferred direction.

The rule is frozen **before** evaluating outcomes and reads no hidden gap
label, goal coordinates or simulator geometry. It is a simple manually
programmed memory baseline, **not** a learned agent. Its usefulness is
demonstrating how far a tiny memory mechanism can go, before attributing
intelligent behaviour to the neural policy.

Both this bit-controller and the existing always-search-up scanner are
evaluated on the SAME held-out world seed/trials and each of the
eight v2 space/time/energy settings. The main practical test is the
tight-time regime in which the always-up scanner previously failed
when the opening was below.

## Pilot invocation

```bash
python -m caesar.memory_causal_probe \
  --seeds 11 23 --train-episodes 50 --probe-trials 12 \
  --control-trials 8 --output outputs/caesar_gru_memory_pilot.json
```

## Interpretation guardrails

- Preserved-history action differences establish only **causal dependence
  on recurrent history at a fixed immediate input**. They do not prove
  the policy retained the right information or used it to navigate.
- Correct two-sided directional action is a much more specific
  criterion. For any claim of learned information use, demonstrate that
  preserved-history **trained** policies outperform same-policy reset
  and untrained baselines over independent seeds.
- A hand-written one-bit cue that works demonstrates the *possibility*
  of a solution from sensor memory, not neural learning.
- If the bit cue fails, measure decoding accuracy rather than
  retuning thresholds to the held-out trials.
- Hard held-out end-to-end success remains the ultimate benchmark;
  success on a forced-approach probe is only an intermediate criterion.
- The original v1 and v2 negative results remain retained unchanged.
