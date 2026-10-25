"""CAESAR v2: fixed observation-only wall scanner and detour-cost audit.

No learning rule, reward, network, geometry, or v1 benchmark is changed.
The scanner sees only the same 14-vector as CAESAR, with finite state
and fixed thresholds specified before evaluating results.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .baselines import controller_action
from .challenge_benchmark import EVAL_SEED_OFFSET, oracle_action
from .challenge_env import BarrierEnv, CHALLENGE_ARMS, challenge_config

CONTROLS = ("direct", "wall_scanner", "oracle_route")


@dataclass
class WallScanner:
    """Fixed policy that follows the visible wall, exploring upward first.

    No global map, gap flag, coordinates, reward or simulator internals:
    only the 14-element CAESAR observation and a four-state finite automaton.
    Thresholds and 15-step wall crossing are frozen *a priori*.
    """
    phase: str = "approach"
    crossing_steps: int = 0
    phase_counts: dict[str, int] = field(default_factory=dict)

    def action(self, obs: np.ndarray) -> np.ndarray:
        if obs.shape != (14,):
            raise ValueError("expected CAESAR's 14-element observation")
        east_ray = float(obs[6])
        north_ray = float(obs[8])
        velocity = obs[0:2]
        goal_direction = obs[2:4]
        if self.phase == "approach" and east_ray < 0.065:
            self.phase = "search_up"
        if self.phase == "search_up":
            if east_ray > 0.095:
                self.phase = "cross"
                self.crossing_steps = 0
            elif north_ray < 0.06:
                self.phase = "search_down"
        if self.phase == "search_down" and east_ray > 0.095:
            self.phase = "cross"
            self.crossing_steps = 0
        if self.phase == "cross":
            self.crossing_steps += 1
            if self.crossing_steps >= 18:
                self.phase = "finish"
        self.phase_counts[self.phase] = self.phase_counts.get(self.phase, 0) + 1
        if self.phase == "search_up":
            desired = np.array([0.0, 1.0], dtype=np.float32)
        elif self.phase == "search_down":
            desired = np.array([0.0, -1.0], dtype=np.float32)
        elif self.phase == "cross":
            desired = np.array([1.0, 0.0], dtype=np.float32)
        else:
            desired = goal_direction.astype(np.float32)
        return np.clip(desired - velocity, -1, 1).astype(np.float32)


def trial_rollout(seed: int, arm: str, trial: int, control: str) -> dict:
    if arm not in CHALLENGE_ARMS or control not in CONTROLS or trial < 0:
        raise ValueError("invalid arm, controller or trial")
    # The same trial setup as the existing frozen learned-policy evaluator.
    rng = np.random.default_rng(EVAL_SEED_OFFSET + seed * 10_000 + trial)
    env = BarrierEnv(config=challenge_config(arm), rng=rng)
    obs = env.reset()
    initial_pos = env.agent_pos.copy()
    goal = env.goal_pos.copy()
    initial_dist = float(np.linalg.norm(initial_pos - goal))
    automaton = WallScanner()
    stage = 0
    done = False
    total_reward = 0.0
    discount = 1.0
    discounted_reward = 0.0
    distance_increase_steps = 0
    wall_crossing_step = None
    max_distance = initial_dist
    first_away_step = None
    first_away_reward = None
    last_distance = initial_dist
    first_crossing_step = None
    while not done:
        old_x = float(env.agent_pos[0])
        if control == "oracle_route":
            action, stage = oracle_action(env, stage)
        elif control == "wall_scanner":
            action = automaton.action(obs)
        else:
            action = controller_action(obs, "direct")
        obs, reward, done, info = env.step(action)
        total_reward += reward
        discounted_reward += discount * reward
        discount *= 0.98
        distance = float(info["distance"])
        max_distance = max(max_distance, distance)
        if distance > last_distance + 1e-5:
            distance_increase_steps += 1
            if first_away_step is None:
                first_away_step = env.steps
                first_away_reward = float(reward)
        last_distance = distance
        if first_crossing_step is None and old_x < env.wall_x <= float(env.agent_pos[0]):
            first_crossing_step = env.steps
    return {
        "trial": trial, "seed": seed, "arm": arm, "control": control,
        "gap_side": env.gap_side,
        "initial_distance": initial_dist,
        "goal_bearing_y": float((goal - initial_pos)[1]),
        "success": int(info["success"]),
        "steps": env.steps,
        "battery_remaining": float(env.battery),
        "collisions": int(env.collision_count),
        "total_return": float(total_reward),
        "discounted_return_0_98": float(discounted_reward),
        "first_wall_crossing_step": first_crossing_step,
        "distance_increase_steps": distance_increase_steps,
        "first_away_step": first_away_step,
        "first_away_reward": first_away_reward,
        "max_distance_excess": float(max_distance - initial_dist),
        "scanner_phase_counts": automaton.phase_counts if control == "wall_scanner" else None,
    }


def run_audit(seeds: list[int], trials: int, arms: list[str]) -> dict:
    if not seeds or len(seeds) != len(set(seeds)) or trials <= 0:
        raise ValueError("need unique seeds and positive trial count")
    if not arms or len(arms) != len(set(arms)) or any(arm not in CHALLENGE_ARMS for arm in arms):
        raise ValueError("invalid or duplicate arms")
    rows = [
        trial_rollout(seed, arm, trial, control)
        for seed in seeds for arm in arms for trial in range(trials)
        for control in CONTROLS
    ]
    summary = {}
    for control in CONTROLS:
        target = [row for row in rows if row["control"] == control]
        summary[control] = {
            "total_trials": len(target),
            "successes": sum(row["success"] for row in target),
            "success_rate": float(np.mean([row["success"] for row in target])),
            "mean_steps": float(np.mean([row["steps"] for row in target])),
            "mean_battery_remaining": float(np.mean([row["battery_remaining"] for row in target])),
            "mean_collisions": float(np.mean([row["collisions"] for row in target])),
            "mean_distance_increasing_steps": float(np.mean([
                row["distance_increase_steps"] for row in target
            ])),
            "mean_max_distance_excess": float(np.mean([
                row["max_distance_excess"] for row in target
            ])),
            "mean_discounted_return": float(np.mean([
                row["discounted_return_0_98"] for row in target
            ])),
            "success_by_gap_side": {
                side: float(np.mean([row["success"] for row in target if row["gap_side"] == side]))
                if any(row["gap_side"] == side for row in target) else None
                for side in ("top", "bottom")
            },
            "per_arm": {
                arm: {
                    "n": len([row for row in target if row["arm"] == arm]),
                    "success_rate": float(np.mean([row["success"] for row in target if row["arm"] == arm])),
                } for arm in arms
            },
        }
    return {
        "protocol": "caesar_v2_detour_cost_and_no_learning_scanner_v1",
        "seeds": seeds, "trials_per_arm_per_seed": trials, "arms": arms,
        "controller_definitions": {
            "direct": "CAESAR observation's goal bearing as bounded action",
            "wall_scanner": (
                "Finite-state controller reading only observation's velocity,"
                " goal bearing, east/north rays; scans upwards first, reverses"
                " at northern wall, crosses when east ray opens for 18 actions;"
                " fixed thresholds and no learning or simulator internals"
            ),
            "oracle_route": (
                "PRIVILEGED geometry-informed waypoint planner, only a "
                "positive physical-feasibility reference, not fair control"
            ),
        },
        "interpretation_limit": (
            "A longer route may temporarily INCREASE Euclidean goal distance, "
            "but this by itself does not establish the reason CAESAR fails. "
            "The shaped reward penalizes all time steps in proportion to distance; "
            "success reward may outweigh the interim negative return. "
            "No learner was trained in this audit."
        ),
        "summary": summary, "trials": rows,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Audit detour costs and frozen no-learning wall scanner")
    p.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 47])
    p.add_argument("--trials", type=int, default=24)
    p.add_argument("--arms", nargs="+", choices=CHALLENGE_ARMS, default=list(CHALLENGE_ARMS))
    p.add_argument("--output", default="outputs/detour_cost_audit.json")
    args = p.parse_args()
    result = run_audit(args.seeds, args.trials, args.arms)
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "protocol": result["protocol"],
        "seeds": args.seeds, "trials": args.trials,
        "arms": args.arms, "summary": result["summary"],
        "interpretation_limit": result["interpretation_limit"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
