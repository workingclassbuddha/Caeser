"""CAESAR v2: does progressively narrowing a passage help learning?

All policy, optimizer, reward and observation code is unchanged.
Only a physical gap width varies during training; final test is identical.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .challenge_env import BarrierEnv, CHALLENGE_ARMS, GAP_HALF_WIDTH, challenge_config
from .challenge_benchmark import evaluate
from .core import CAESARNet
from .training import (
    TrainingConfig, apply_actor_critic_update, compute_noise_scale,
    run_training_episode, set_seed,
)

SCHEDULES = ("fixed_hard", "graduated")
WIDE_HALF_WIDTH = 2.1
HARD_HALF_WIDTH = GAP_HALF_WIDTH
FULL_DIFFICULTY_FRACTION = 0.70


def width_for_episode(schedule: str, episode: int, episodes: int) -> float:
    if schedule not in SCHEDULES or episodes < 1 or not 0 <= episode < episodes:
        raise ValueError("invalid schedule or episode")
    if schedule == "fixed_hard":
        return HARD_HALF_WIDTH
    progress = min(1.0, episode / max(1.0, episodes * FULL_DIFFICULTY_FRACTION))
    return WIDE_HALF_WIDTH - (WIDE_HALF_WIDTH - HARD_HALF_WIDTH) * progress


def train_schedule(seed: int, schedule: str, episodes: int, trials: int) -> dict:
    if schedule not in SCHEDULES or episodes < 1 or trials < 1:
        raise ValueError("invalid schedule, episode count or trial count")
    set_seed(seed)
    env = BarrierEnv(config=challenge_config("none"), rng=np.random.default_rng(seed))
    agent = CAESARNet()
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    config = TrainingConfig(seed=seed, device="cpu", episodes=episodes)
    before = {arm: evaluate(seed, arm, trials, controller=None, agent=agent)
              for arm in CHALLENGE_ARMS}
    episodes_log = []
    for index in range(episodes):
        env.gap_half_width = width_for_episode(schedule, index, episodes)
        rollout = run_training_episode(
            env, agent, torch.device("cpu"), compute_noise_scale(index, config)
        )
        policy_loss, value_loss = apply_actor_critic_update(
            agent, optimizer, rollout, config, torch.device("cpu")
        )
        episodes_log.append({
            "episode": index, "gap_half_width": env.gap_half_width,
            "success": int(rollout.success), "steps": rollout.steps,
            "return": rollout.total_reward,
            "scar": float(agent.scar.item()), "mirror": float(agent.M_state.item()),
            "policy_loss": policy_loss, "value_loss": value_loss,
        })
    after = {arm: evaluate(seed, arm, trials, controller=None, agent=agent)
             for arm in CHALLENGE_ARMS}
    last = episodes_log[-max(1, episodes // 3):]
    return {
        "seed": seed, "schedule": schedule, "episodes": episodes,
        "overall_training_success": float(np.mean([r["success"] for r in episodes_log])),
        "last_third_training_success": float(np.mean([r["success"] for r in last])),
        "ending_width": episodes_log[-1]["gap_half_width"],
        "pretrain": before, "posttrain": after,
        "training_trace": episodes_log,
    }


def run_curriculum(seeds: list[int], episodes: int, trials: int) -> dict:
    if not seeds or len(set(seeds)) != len(seeds) or episodes < 2 or trials < 1:
        raise ValueError("Need unique seeds, at least two episodes, positive trials")
    torch.set_num_threads(1)
    runs = [train_schedule(seed, schedule, episodes, trials)
            for seed in seeds for schedule in SCHEDULES]
    summary = {}
    for schedule in SCHEDULES:
        group = [r for r in runs if r["schedule"] == schedule]
        summary[schedule] = {
            "training_success": float(np.mean([
                r["overall_training_success"] for r in group
            ])),
            "last_third_training_success": float(np.mean([
                r["last_third_training_success"] for r in group
            ])),
            "hard_test_success": float(np.mean([
                r["posttrain"]["none"]["success_rate"] for r in group
            ])),
            "hard_test_delta_from_untrained": float(np.mean([
                r["posttrain"]["none"]["success_rate"] -
                r["pretrain"]["none"]["success_rate"] for r in group
            ])),
            "all_eight_hard_test_conditions_success": float(np.mean([
                r["posttrain"][arm]["success_rate"]
                for r in group for arm in CHALLENGE_ARMS
            ])),
            "per_training_seed_hard_test_success": {
                str(r["seed"]): r["posttrain"]["none"]["success_rate"] for r in group
            },
        }
    baselines = {
        str(seed): {
            "direct": evaluate(seed, "none", trials, controller="direct"),
            "oracle_feasibility": evaluate(seed, "none", trials, controller="oracle_route"),
        } for seed in seeds
    }
    return {
        "protocol": "caesar_v2_progressive_physical_constraint_v1",
        "seeds": seeds, "episodes": episodes, "trials": trials,
        "schedules": list(SCHEDULES),
        "wide_half_width": WIDE_HALF_WIDTH,
        "final_hard_half_width": HARD_HALF_WIDTH,
        "hard_environment_from_training_fraction": FULL_DIFFICULTY_FRACTION,
        "description": (
            "Matched-seed actor-critic learner and reward, no solution "
            "waypoints supplied. Fixed-hard wall opening versus an opening "
            "progressively narrowed in early training. Both evaluated "
            "without policy updates on identical hard-gap layouts."
        ),
        "runs": runs, "baselines": baselines, "summary": summary,
        "limits": (
            "Short runs are exploratory. A curriculum is additional "
            "training environment design, not zero prior structure. "
            "Proving a general learning principle needs strong "
            "observation-only controls and many independent seeds."
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Fixed-hard vs progressive gap curriculum")
    p.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 47])
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--output", default="outputs/challenge_curriculum.json")
    args = p.parse_args()
    report = run_curriculum(args.seeds, args.episodes, args.trials)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "protocol": report["protocol"],
        "seeds": report["seeds"], "episodes": report["episodes"],
        "trials": report["trials"], "summary": report["summary"],
        "caution": report["limits"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
