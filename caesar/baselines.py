"""Nonlearning goal-following controls for CAESAR constraint-transfer tests.

Uses ONLY observations available to CAESAR (velocity and goal direction).
Controller coefficients are fixed before evaluation, with no training/tuning.
Test seeds and environment construction match constraint_transfer.evaluate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .constraint_transfer import ARMS, arm_config

CONTROLLERS = ("direct", "velocity_compensated", "stationary")


def controller_action(observation: np.ndarray, controller: str) -> np.ndarray:
    """Fixed nonlearning policies, bounded by the same [-1, 1] action limits."""
    if controller not in CONTROLLERS:
        raise ValueError(f"unknown controller: {controller}")
    if observation.shape != (14,):
        raise ValueError("expected the standard 14-dimensional CAESAR observation")
    direction = observation[2:4].astype(np.float32)
    velocity = observation[0:2].astype(np.float32)
    if controller == "stationary":
        action = np.zeros(2, dtype=np.float32)
    elif controller == "direct":
        action = direction
    else:
        # Fixed one-step velocity damping, no gains selected using test results.
        action = direction - velocity
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def evaluate_controller(controller: str, test_arm: str, seed: int, trials: int) -> dict:
    """Deterministic trial episodes, paired with the learned-agent evaluation."""
    from .core import EmbodiedEnv

    if controller not in CONTROLLERS:
        raise ValueError(f"unknown controller: {controller}")
    if trials <= 0:
        raise ValueError("trials must be positive")
    if test_arm not in ARMS:
        raise ValueError(f"unknown arm: {test_arm}")
    successes, steps, terminal_battery, returns = [], [], [], []
    for trial in range(trials):
        env = EmbodiedEnv(
            level=0, config=arm_config(test_arm),
            rng=np.random.default_rng(100_000 + seed * 10_000 + trial),
        )
        # Match the learned evaluator's explicit reset after construction.
        observation = env.reset()
        done = False
        episode_return = 0.0
        while not done:
            action = controller_action(observation, controller)
            observation, reward, done, info = env.step(action)
            episode_return += float(reward)
        successes.append(int(info["success"]))
        steps.append(env.steps)
        terminal_battery.append(float(info["battery"]))
        returns.append(episode_return)
    return {
        "success_rate": float(np.mean(successes)),
        "successes": int(sum(successes)),
        "trials": trials,
        "mean_steps": float(np.mean(steps)),
        "mean_terminal_battery": float(np.mean(terminal_battery)),
        "mean_return": float(np.mean(returns)),
        "trial_success": successes,
    }


def run_baselines(seeds: list[int], trials: int) -> dict:
    if not seeds or len(set(seeds)) != len(seeds) or trials <= 0:
        raise ValueError("nonempty unique seeds and positive trials required")
    evaluations: dict[str, dict[str, dict[str, dict]]] = {}
    summary = {}
    for controller in CONTROLLERS:
        evaluations[controller] = {}
        for seed in seeds:
            evaluations[controller][str(seed)] = {
                arm: evaluate_controller(controller, arm, seed, trials) for arm in ARMS
            }
        summary[controller] = {
            "overall_test_success": float(np.mean([
                evaluations[controller][str(seed)][arm]["success_rate"]
                for seed in seeds for arm in ARMS
            ])),
            "test_success_by_arm": {
                arm: float(np.mean([
                    evaluations[controller][str(seed)][arm]["success_rate"]
                    for seed in seeds
                ])) for arm in ARMS
            },
            "cross_condition_success_excluding_training_arm": {
                train_arm: float(np.mean([
                    evaluations[controller][str(seed)][test_arm]["success_rate"]
                    for seed in seeds for test_arm in ARMS if test_arm != train_arm
                ])) for train_arm in ARMS
            },
        }
    return {
        "protocol": "caesar_nonlearning_observation_controller_v1",
        "seeds": seeds,
        "heldout_trials_per_test_arm": trials,
        "eval_arms": list(ARMS),
        "controllers": list(CONTROLLERS),
        "comparison": (
            "Uses the exact same post-construction reset and held-out seed formula "
            "as constraint_transfer.evaluate. All controllers use only the "
            "14-dimensional observation, no oracle state and no learning. "
            "The pilot has not shown superiority of a learned policy."
        ),
        "evaluations": evaluations,
        "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate no-learning CAESAR goal controllers")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 47])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--output", default="outputs/controller_baselines.json")
    args = parser.parse_args()
    report = run_baselines(args.seeds, args.trials)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "protocol": report["protocol"],
        "seeds": report["seeds"],
        "trials": report["heldout_trials_per_test_arm"],
        "summary": report["summary"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
