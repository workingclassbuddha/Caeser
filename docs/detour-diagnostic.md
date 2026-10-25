# CAESAR v2 detour / one-step incentive diagnostic (2026-10-25)

This module exists because the v2 agent can reach a valid target only
by passing an opening in a wall, and the unchanged recurrent learner
has not solved the held-out challenge after the recorded budgets. In the
first physical curriculum pilot, successful *training* in wide-gap
conditions did **not** carry over to the hard environment.

## Falsifiable questions

1. Does a successful **reference** trajectory require at least some
   individual actions which reduce Euclidean goal distance *less* than
   direct goal-following would at the exact same state?
2. Does a fixed, **observation-only** wall-scanner policy (with no
   learning) beat the existing direct-to-goal policies when the gap is
   randomly top or bottom? If yes, how far can even a four-state
   hand-written controller get?
3. How do the actual route duration, battery costs, collisions and
   discounted returns differ between successful detours and failed
   direct navigation?

These tests **do not** attempt to train a new learner or prove a mechanism.
They are diagnostics on the environment and a stronger non-learning
control before we tune the reward or the neural architecture.

## Interventions and controls

- **Direct steering:** fixed v1 goal-direction controller, from the
  standard 14-vector (goal bearing is openly observed).
- **Observation-only wall scanner:** fixed four/five phase state-machine
  with `approach`, `search_up`, `search_down`, `cross`, `finish`.
  It approaches the wall, scans upward using the east/north range rays,
  reverses when the north ray shows a nearby boundary, moves east when
  the ray opens, then follows goal direction. It uses only
  goal direction, velocity and local range values, plus its own phase
  counter. All thresholds and crossing duration fixed *before*
  seeing held-out results. It knows no gap side or wall coordinates.
- **Privileged reference:** existing PD waypoint controller using
  simulator gap geometry (already in v2). **Never** classify it as
  an equal-observation baseline.

The physical environment, CAESAR policy, optimizer, time/energy
mechanics and shaped reward are not modified.

## Same-state one-step counterfactual

For each actual trajectory step, copy the simulator state at that
instant and test the alternative action equal to direct observed goal
steering. Record:

```text
distance_sacrifice = distance_after_actual_action
                     - distance_after_direct_action_from_same_state
```

Positive values demonstrate a genuine *short-term geometric tradeoff*
on that step, even if the detour's distance never exceeds its **initial**
distance to the goal. These replayed direct actions are only
**measurements**, never rewards/observations fed to the agent.

Also log the number of steps where actual goal distance rises, final
battery, collisions, steps to first wall crossing, cumulative reward,
and the discounted sum using the existing 0.98 return factor. Not
every one-step sacrifice means a strictly worse **reward**, because
velocity, battery, terminal bonuses and step count also matter.
This audit does not attribute causal learning failure to the reward;
credit assignment is a hypothesis needing a controlled intervention.

## Exact pilot command

```bash
python -m caesar.detour_diagnostic --seeds 11 23 47 \
    --trials 4 \
    --output outputs/detour_cost_pilot.json
```

Every one of the eight v2 test constraint arms is evaluated with
the same withheld starts/goals per seed/trial/control. The positive
reference route is used only to confirm feasibility.

## What qualifies as progress

- A successful reference route with positive one-step geometric
  opportunity costs would establish that the hard task does require
  actions unlike the immediate goal-following reaction.
- A no-learning wall scanner achieving any positive held-out
  performance establishes another baseline the learned policy should
  exceed before claiming its route-finding advantage.
- If the scanner also fails, it reveals limits of this simple strategy,
  *not* proof that neural learning is necessary.
- Any comparison of trained agents to this controller must use exactly
  the same held-out seeds/configurations and adequate training. Nothing
  in this audit supports consciousness or a new general intelligence.
