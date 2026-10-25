# CAESAR v2: can a frozen GRU contain a useful early sensory clue?

**Predeclared 25 Oct 2026.** The prior v2 causal study confirmed:
- Top-vs-bottom openings are distinguishable by early ray sensors, but
  almost never by the current ray observations just before the wall.
- A frozen one-bit controller achieved 364/384 (94.8%) on new routes.
- Two CAESAR GRUs trained for 50 episodes each showed history-dependent
  actions but chose the correct *opposite* directions in 0/22 matched
  top/bottom observation pairs.

The causal alternatives now are: (A) opening information no longer exists
in CAESAR's recurrent hidden representation; (B) it is present but its
actor readout does not use it to choose correctly. Decoding information is
not equivalent to learning a decision, and an **untrained GRU may encode**
early clues without learning anything.

## Frozen methods

For each model seed (11 and 23 in pilot), sample two independent groups of
top/bottom world pairs:
- Decoder **fitting**: world seeds 31,37,41 (24 pairs each initially).
- Decoder **holdout**: world seeds 89,101,211 (24 pairs each initially).
- Seed namespace is separate from on-policy training and test positions.
- Both worlds of a pair have identical start and goal, but an upper vs lower
  wall opening. Force the *same physical approach actions* until near the
  wall. Retain only examples whose final 14-dimensional observation vectors
  match with absolute difference below 1e-6.
- Sample labels are balanced: every eligible pair adds one 'top' and one
  'bottom' example. Keep pairs together across train/test; **never split
  samples from a pair across sets**.
- With the same frozen CAESAR network, extract the hidden state after
  processing the observation history, and the hidden state when reset to
  zero just before the identical final observation. Model buffers are
  saved/restored as in PR #15.
- Compare separately: **early observation**, **current observation**,
  **history-preserved GRU hidden state**, **last-step-only/reset GRU hidden**.

A single fixed **ridge linear decoder, alpha=25.0**, is fitted with known
top/bottom labels only on the fitting worlds and evaluated on held-out
worlds. Standardize features using only fitting-set mean/variance,
and never tune hyperparameters based on holdout scores. Include a
deterministically shuffled-training-label control.

For each policy seed, compare the random-weight network to the
**same network after unchanged actor-critic training** for 50 episodes.
Return per-pair hidden vectors, fitted decoder predictions, gap side,
world seed, successful training episode count, and held-out accuracies.

## Positive and negative evidence

- If current observation and reset-hidden decoders are ~50% on exactly
  matched pairs, that confirms information cannot come from the
  instantaneous input in this controlled contrast.
- If **untrained** preserved-history hidden states decode well, then
  temporal information is present in random recurrent features; no
  learning claim follows.
- If trained preserved-history hidden states decode better than
  untrained hidden states on independent world seeds, that suggests
  training changed usable representation, but with only two training
  seeds is still a descriptive result, not robust causal evidence.
- If trained hidden state is highly decodable yet actor sign correctness
  remains zero, the specific bottleneck lies **downstream of information
  retention**, somewhere between policy readout and learned action
  preference. An external supervised decoder adds knowledge the
  original agent did not possess.
- No result here by itself establishes conscious memory or general
  intelligence. Demonstrating useful learned memory requires a
  held-out action/route benefit with history preserved versus erased.

## Pilot (engineering)

```bash
python -m caesar.gru_information_decoder \
  --policy-seeds 11 23 --train-episodes 50 \
  --train-trials 24 --test-trials 24 \
  --output outputs/gru_information_pilot.json
```

The full per-seed raw features and predictions are retained for auditing.
Any follow-up with longer training, curriculum or new reward learning
rules must be separately predeclared to avoid interpreting hindsight
selection as a discovery.
