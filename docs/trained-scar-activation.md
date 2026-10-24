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

## Repeated training-reachability scan

The CI also runs a preselected 3-seed, 20-episode level-0 scan:

```bash
python -m caesar.trained_scar_probe --seeds 11 23 47 --episodes 20 \
  --level 0 --output outputs/trained_scar_scan.json
```

The expanded sample evaluates **whether scar activates during training**
rather than assuming that a paired ablation is meaningful. A positive finding
would justify a subsequent pre-registered paired learning study with held-out
environments; a zero finding means the current stock setup lacks demonstrated
scar-memory exposure, not that activation is impossible under all conditions.
