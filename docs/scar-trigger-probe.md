# Scar activation pre-test

Before interpreting CAESAR's paired scar/no-scar comparison, check whether the
stock agent enters a state where scar can increment at all.

```bash
python -m caesar.scar_probe --episodes 3 --seed 42 --level 0
python -m caesar.scar_probe --episodes 3 --seed 42 --level 10
```

The diagnostic uses seeded, untrained deterministic policy actions and the
actual model forward pass. It reports battery, mirror and the number of
increment events. This is **not** a substitute for learning experiments or
held-out evaluation, and should not be presented as a neuroscience result.

Scar increments only when **both** `mirror < scar_threshold` and
`battery < scar_battery_threshold` on a forward call. A scar that never
increments makes any no-scar training ablation uninformative about hysteresis.

If no triggers occur:
1. Record the null finding and full configuration first.
2. Run a separate pre-registered stress manipulation to test reachability.
3. Avoid tuning the default environment until a positive result appears;
   that would make the baseline comparison post-hoc.
4. Track task outcomes independently of the internal mirror and scar.
