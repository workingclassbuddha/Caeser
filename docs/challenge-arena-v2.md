# CAESAR Challenge Arena v2: learning under meaningful constraints

This is **an entirely new, versioned task**, not a revision of the
original benchmark. The v1 failure is permanent evidence: two hand-written
observation-only controllers solved all 720 held-out v1 goals each, whereas
the lightly trained neural agent did worse. These results must not be
overwritten or reframed as proof of learned intelligence.

## A world with a physical obstacle

- A vertical, impenetrable wall divides a 5×5 environment into two regions.
- Each episode places the opening at the **top or bottom**, randomly from
  the episode's seed. The agent starts on the left near mid-height, the goal
  on the right near mid-height. Following the goal bearing hits the wall.
- The original action physics, reward equations, GRU policy, gradients and
  trainer are unchanged; colliding means motion is clamped and energy/time
  are still consumed. No additional collision penalty or hand-written route
  reward is provided.
- The existing 8 world-oriented range sensors now detect the dividing wall
  as well as the arena boundary; velocity, current goal direction, relative
  goal distance and energy retain the same 14 observation slots.
- The policy is **never given** the gap side or a route map. The privileged
  fixed-waypoint controller knows the gap location for positive feasibility
  testing only, and is not a fair learned-vs-controller baseline.

## Constraint interventions (2³ factorial)

| Tightened factor | Standard v2 | Tightened v2 |
| --- | --- | --- |
| Space / precision of goal | radius 0.50 | radius 0.30 |
| Time / maximum actions | 190 | 105 |
| Energy / thrust cost | 0.15 | 0.36 |

Every arm has the same wall, target distribution, battery capacity 50,
velocity dynamics, distance/time reward and remaining-energy success
bonus. No factor turns off the physical world.

Eight arms correspond to every combination of these three limits.

## Controls and outcomes

Precommitted no-learning controls are direct goal pointing, goal pointing
minus observed velocity, and stationary action. They receive exactly the
observations supplied to the network, not environment attributes.

The *privileged reference route* uses wall opening and goal coordinates
to test whether the challenge is physically feasible. Its success rate is
not a trained-policy comparison and must never be marketed as an
observation-only baseline.

For each arm and independent seed, report:
- Train history: each episode's success, return, steps, scar, mirror and losses
- Held-out before/after **frozen-weight** success, steps, energy left,
  wall collisions, split by gap side, and all eight test constraint arms
- Exact same held-out trial seeds for all controllers and trained agents
- Whether the privileged reference route is successful in the tightened
  conditions; never claim the task is learnable if no feasible path exists

## Precommitted smoke test, not a discovery

```bash
python -m caesar.challenge_benchmark --seeds 11 \
  --episodes 12 --trials 3 \
  --output outputs/challenge_v2_pilot.json
```

The pilot exercises **eight training arms**, but only one learning seed
and 12 gradient updates per arm. It is an engineering smoke. A 0% learned
success rate here does not tell us what would happen with sufficient
training; conversely a positive rate can arise from luck or learned
simple behaviour without generality.

## Next decision gates

1. Verify collision cannot be bypassed and gap opening permits crossing.
2. Confirm direct goal pointing now fails a nontrivial proportion of
   held-out trials and the privileged reference consistently succeeds
   within time/energy budgets.
3. Fix any physical impossibility or solver bug **before** evaluating learned
   performance, and document all changes as exploratory.
4. Run at least five independent training seeds with long enough training
   for meaningful adaptation, and test disjoint unseen gap configurations.
5. If long-run training still cannot beat a simple non-learning controller,
   record a negative result. Don't respond by adding unmeasured mental
   mechanisms or changing the score definition.

This is a task of learning under physical constraints, not evidence of
consciousness or a universal theory of intelligence.
