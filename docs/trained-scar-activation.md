# Scar activation during actual training

The earlier stock-environment probes used an untrained policy and deterministic
action means. They found 0 activation events in the tested conditions, but
could not establish what happens during actor-critic training.

This diagnostic performs **real gradient updates** while keeping the stock
CAESAR environment, model, reward and scar thresholds unchanged.

```bash
python -m caesar.trained_scar_probe --seed 42 --episodes 5 --level 0
```

It records episode-level policy/value losses, outcomes, battery at termination,
mirror state, and scar increments estimated from the change in the monotonically
accumulating scar buffer. Raw output goes to
`outputs/trained_scar_probe.json`.

The CI smoke run uses five episodes. This is deliberately shorter than the
3,000-episode research programme: its only purpose is to establish that the
training path can run and collect a few real measurements.

**Interpretation:** A zero result is a measured absence of activation over the
sampled episodes, not proof of impossibility. A positive result is evidence of
activation, *not evidence of improved learning*. Only then is a paired
scar/no-scar ablation with held-out outcomes informative.
