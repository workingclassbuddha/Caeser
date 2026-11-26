# CAESAR scar mechanism: source-level causal audit (2026-10-24)

## What the current code actually implements

The recurrent GRU, actor and critic use the **input observation** and
**within-episode recurrent hidden state**. The model stores scalar `scar`
and `M_state` buffers, but neither is concatenated with the observation nor
fed into the actor/critic during `CAESARNet.forward`.

The mirror state and low-battery test can **increase** the persistent scar
buffer. The training loop then uses that buffer in:

```python
health_factor = 1.0 / (agent.scar.item() + 0.1)
policy_loss = -(log_probs * advantages * health_factor).mean()
```

Consequently scar can alter **gradient updates and future network weights**.
With a fixed set of weights, deliberately changing only `scar` does not
change the network's current action, value prediction or GRU hidden output.
A regression test now verifies this directly.

This distinction matters: the existing code does not establish a
cross-episode *memory used for action selection*. It implements a persistent
training-modulation signal. The within-episode GRU itself is recurrent memory.

## What experiments showed

- Stock untrained: 10 runs at 5 seeds × levels 0/10, zero scar increments.
- Stock trained: 3 seeds × 20 episodes = 14,966 steps, zero scar increments.
- Exploratory, post-hoc long horizon: one seed, 5 trained episodes,
  3,118 steps, 109 scar increments, final scar ~0.000545.

The stress result shows the gate can activate when the episode horizon is
extended; it does **not** show that doing so improves learning.

At scar=0, `health_factor` is 10. At the observed stress scar=0.000545,
it becomes about 9.9458 (roughly 0.54% lower). Gradient clipping, value-loss
weighting and optimizer dynamics may further change the effective effect
size; that effect has not been isolated experimentally.

## Next falsifiable test

Use matched-seed training pairs **under a fixed, explicitly labelled
extended-horizon environment**, with scar increment enabled versus zero.
Keep all other model/environment parameters and the data budget identical.
Inspect actual scar exposure, the downstream policy-loss changes, and
held-out task success/generalization. A negative or null performance
difference is a valid result and argues against a practical benefit.

Do not claim a new kind of memory, consciousness, clinical benefit, or
universal dynamics based on the existence of a scalar scar buffer.
