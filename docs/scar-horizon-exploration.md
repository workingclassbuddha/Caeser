# Post-hoc exploratory test: is the scar gate reachable with a longer horizon?

**This test was designed after finding zero scar increments in the original
stock training runs. It is exploratory and cannot validate a preregistered
prediction about stock CAESAR behaviour.**

The stock environment terminates after 350 steps at level 0 unless the goal or
battery limit terminates sooner. Scar requires *both* mirror below 0.12 and
battery below 5.0 during a forward call. The short horizon may prevent these
conditions from coinciding.

A controlled stress arm changes **only the episode time limit**, not the model,
training rule, scar threshold, initial battery or energy drain.

```bash
python -m caesar.trained_scar_probe --seed 42 --episodes 5 --level 0 \
    --max-steps 1500 --output outputs/trained_scar_extended_horizon.json
```

The normal-environment controls remain in CI and must be reported separately.

### Questions

1. Does training with the longer horizon ever produce joint low-battery and
   low-mirror conditions before the environment terminates?
2. Does the scar buffer actually accumulate as a result?
3. If zero, does the problem lie in the gate itself, even after permitting longer
   trajectories? A separate, registered stress protocol would then be needed.

A positive answer to questions 1 or 2 would **only demonstrate reachability**,
not that memory improves learning. That would require an independent paired
training and held-out generalization experiment.
