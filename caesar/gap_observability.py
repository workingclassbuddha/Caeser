"""Can CAESAR's 14 observations reveal which side holds the wall opening?

Construct paired worlds with IDENTICAL start, goal, velocity, battery and
external coordinates; vary only hidden top/bottom opening. Distinguishable
observations identify available sensor information, NOT learned inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from .baselines import controller_action
from .challenge_env import BarrierEnv, challenge_config

OBS_SEED_OFFSET = 33_000_000
EPSILON = 1e-6


def pair(seed: int, trial: int, arm: str = "none") -> dict:
    if trial < 0:
        raise ValueError("trial must be nonnegative")
    pair_seed = OBS_SEED_OFFSET + 10_000 * seed + trial
    cases = {}
    for side in ("top", "bottom"):
        env = BarrierEnv(
            config=challenge_config(arm),
            rng=np.random.default_rng(pair_seed),
            fixed_gap_side=side,
        )
        obs = env.reset()
        cases[side] = (env, obs)
    top, top_obs = cases["top"]
    bottom, bottom_obs = cases["bottom"]
    if not np.array_equal(top.agent_pos, bottom.agent_pos):
        raise AssertionError("paired starts not equal")
    if not np.array_equal(top.goal_pos, bottom.goal_pos):
        raise AssertionError("paired goals not equal")

    observations = [(top_obs, bottom_obs, "spawn", 0)]
    # Goal-directed approach only until the east ray is close to the wall;
    # paired physical motion is identical because the goal bearing and
    # dynamics are the same. No contact or hidden gap label supplied to policy.
    for step in range(1, 100):
        if float(top_obs[6]) < 0.065:
            break
        action = controller_action(top_obs, "direct")
        top_obs, _, top_done, _ = top.step(action)
        bottom_obs, _, bottom_done, _ = bottom.step(action)
        if not np.allclose(top.agent_pos, bottom.agent_pos, atol=1e-6):
            raise AssertionError("paired physical states diverged during approach")
        if top_done or bottom_done:
            break
    observations.append((top_obs, bottom_obs, "near_wall", top.steps))

    def extract(top_o, bottom_o, stage: str, steps: int):
        delta = np.abs(top_o - bottom_o)
        ray_delta = delta[6:14]
        return {
            "stage": stage, "approach_steps": steps,
            "top_sensor_rays": [float(x) for x in top_o[6:14]],
            "bottom_sensor_rays": [float(x) for x in bottom_o[6:14]],
            "nonray_max_delta": float(max(delta[:6])),
            "ray_max_delta": float(max(ray_delta)),
            "ray_l1_delta": float(sum(ray_delta)),
            "informative_rays": [
                int(i) for i,x in enumerate(ray_delta) if x > EPSILON
            ],
            "distinguishable_exact": bool(np.max(ray_delta) > EPSILON),
            "distinguishable_0_01": bool(np.max(ray_delta) > 0.01),
        }

    return {
        "seed": seed, "trial": trial, "arm": arm,
        "initial_position": [float(x) for x in top.agent_pos],
        "target": [float(x) for x in top.goal_pos],
        "observations": [
            extract(top_o, bottom_o, stage, steps)
            for top_o, bottom_o, stage, steps in observations
        ],
    }


def run_audit(seeds: list[int], trials: int, arm: str = "none") -> dict:
    if not seeds or len(seeds) != len(set(seeds)) or trials < 1:
        raise ValueError("Require distinct seeds and positive trials")
    result = [pair(seed, trial, arm) for seed in seeds for trial in range(trials)]
    def describe(stage: str) -> dict:
        rows = [
            next(x for x in test["observations"] if x["stage"] == stage)
            for test in result
        ]
        return {
            "paired_worlds": len(rows),
            "fraction_sensor_distinguishable_exact": float(np.mean([
                x["distinguishable_exact"] for x in rows
            ])),
            "fraction_sensor_distinguishable_at_least_0_01": float(np.mean([
                x["distinguishable_0_01"] for x in rows
            ])),
            "mean_max_ray_difference": float(np.mean([
                x["ray_max_delta"] for x in rows
            ])),
            "nonray_values_match": all(x["nonray_max_delta"] <= EPSILON for x in rows),
            "informative_ray_counts": {
                str(i): sum(i in x["informative_rays"] for x in rows)
                for i in range(8)
            },
        }
    return {
        "protocol": "caesar_v2_matched_world_gap_observability_v1",
        "seeds": seeds, "trials_per_seed": trials, "arm": arm,
        "paired_worlds": len(result),
        "method": (
            "Paired physically identical start-goal states with only hidden gap "
            "side different. Compare the 14 CAESAR observations at spawn and "
            "before wall contact under identical goal-directed approach. "
            "Only 8 ray measurements should differ. No policy learning. "
            "Exact sensor discrimination shows information is AVAILABLE, "
            "not that a learner extracts it or that it suffices to navigate."
        ),
        "summary": {"spawn": describe("spawn"), "near_wall": describe("near_wall")},
        "trials": result,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit hidden gap-side information in CAESAR rays")
    parser.add_argument("--seeds", type=int, nargs="+", default=[89, 101, 211])
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--arm", default="none")
    parser.add_argument("--output", default="outputs/gap_observability.json")
    args = parser.parse_args()
    report = run_audit(args.seeds, args.trials, args.arm)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "protocol": report["protocol"], "arm": args.arm,
        "paired_worlds": report["paired_worlds"],
        "summary": report["summary"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
