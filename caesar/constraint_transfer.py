"""CAESAR: controlled space/time/energy training and cross-condition evaluation.

Factorial intervention: same recurrent actor-critic and reward code in all arms.
This tests constraint-conditioned task learning; it is not a consciousness test.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from .core import CAESARNet, EmbodiedEnv, EnvConfig
from .training import (
    TrainingConfig, apply_actor_critic_update, compute_noise_scale,
    run_training_episode, set_seed,
)

# Frozen intervention definitions, relative to stock level 0.
# min_spawn_distance=1.5 in EVERY condition keeps the sampled initial
# position/goal pairs identical when evaluating with matched RNG seeds.
FACTOR_NAMES = ("space", "time", "energy")
ARMS = (
    "none", "space", "time", "energy",
    "space+time", "space+energy", "time+energy", "space+time+energy",
)
BASE = replace(EnvConfig(), min_spawn_distance=1.5)


def arm_config(arm: str) -> EnvConfig:
    """Change only named physical limits; preserve task and learning rule."""
    if arm not in ARMS:
        raise ValueError(f"Unknown constraint arm: {arm}")
    factors = set() if arm == "none" else set(arm.split("+"))
    return replace(
        BASE,
        initial_success_radius=0.45 if "space" in factors else BASE.initial_success_radius,
        success_radius_floor=0.45 if "space" in factors else BASE.success_radius_floor,
        base_max_steps=150 if "time" in factors else BASE.base_max_steps,
        thrust_battery_cost=0.60 if "energy" in factors else BASE.thrust_battery_cost,
    )


def evaluate(
    agent: CAESARNet, arm: str, seed: int, trials: int,
) -> dict:
    """Deterministic actor-mean policy on unseen trials; no gradient updates.

    Preserve learned scar and mirror buffers to avoid test-environment leakage.
    This evaluates fixed policy competence, not continual learning.
    """
    if trials < 1:
        raise ValueError("trials must be positive")
    was_training = agent.training
    agent.eval()
    saved_scar, saved_mirror = agent.scar.clone(), agent.M_state.clone()
    successes, steps, rewards, energies = [], [], [], []
    config = arm_config(arm)
    try:
        with torch.no_grad():
            for i in range(trials):
                # Identical seed per trial for every training/evaluation arm.
                env = EmbodiedEnv(
                    level=0, config=config,
                    rng=np.random.default_rng(100_000 + seed * 10_000 + i),
                )
                observation = env.reset()
                hidden = torch.zeros(1, 1, agent.hidden_size)
                episode_reward = 0.0
                agent.scar.copy_(saved_scar)
                agent.M_state.copy_(saved_mirror)
                done = False
                while not done:
                    obs = torch.as_tensor(observation, dtype=torch.float32).reshape(1, 1, -1)
                    action, _, hidden = agent(obs, hidden, env.battery)
                    observation, reward, done, info = env.step(
                        action.reshape(-1).cpu().numpy()
                    )
                    episode_reward += reward
                successes.append(int(info["success"]))
                steps.append(env.steps)
                energies.append(float(info["battery"]))
                rewards.append(float(episode_reward))
    finally:
        with torch.no_grad():
            agent.scar.copy_(saved_scar)
            agent.M_state.copy_(saved_mirror)
        agent.train(was_training)
    return {
        "success_rate": float(np.mean(successes)),
        "successes": int(sum(successes)),
        "trials": trials,
        "mean_steps": float(np.mean(steps)),
        "mean_terminal_battery": float(np.mean(energies)),
        "mean_return": float(np.mean(rewards)),
        "trial_success": successes,
    }


def run_arm(seed: int, train_arm: str, episodes: int, eval_trials: int) -> dict:
    if episodes < 1 or eval_trials < 1:
        raise ValueError("episodes and eval_trials must be positive")
    set_seed(seed)
    agent = CAESARNet()
    agent.train()
    config = TrainingConfig(seed=seed, device="cpu", episodes=episodes)
    env = EmbodiedEnv(
        level=0, config=arm_config(train_arm),
        rng=np.random.default_rng(seed),
    )
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate)
    start = {arm: evaluate(agent, arm, seed, eval_trials) for arm in ARMS}
    log = []
    for episode in range(episodes):
        before = float(agent.scar.item())
        rollout = run_training_episode(
            env, agent, torch.device("cpu"), compute_noise_scale(episode, config)
        )
        policy_loss, value_loss = apply_actor_critic_update(
            agent, optimizer, rollout, config, torch.device("cpu")
        )
        log.append({
            "episode": episode,
            "success": int(rollout.success),
            "steps": rollout.steps,
            "return": rollout.total_reward,
            "scar": float(agent.scar.item()),
            "scar_delta": float(agent.scar.item()) - before,
            "mirror": float(agent.M_state.item()),
            "policy_loss": policy_loss,
            "value_loss": value_loss,
        })
    end = {arm: evaluate(agent, arm, seed, eval_trials) for arm in ARMS}
    return {
        "seed": seed,
        "train_arm": train_arm,
        "train_config": asdict(arm_config(train_arm)),
        "train_episodes": episodes,
        "train_success_rate": float(np.mean([row["success"] for row in log])),
        "train_last_10_success_rate": float(np.mean([row["success"] for row in log[-10:]])),
        "train_steps": sum(row["steps"] for row in log),
        "final_scar": float(agent.scar.item()),
        "start_eval": start,
        "end_eval": end,
        "log": log,
    }


def run_study(
    seeds: list[int], episodes: int, eval_trials: int, arms: list[str] | None = None
) -> dict:
    arms = list(ARMS) if arms is None else list(arms)
    if not seeds or len(seeds) != len(set(seeds)) or not arms or len(arms) != len(set(arms)):
        raise ValueError("Seeds and training arms must be nonempty with no duplicates")
    if any(arm not in ARMS for arm in arms):
        raise ValueError("Unknown constraint arm")
    if episodes < 1 or eval_trials < 1:
        raise ValueError("Episode/trial counts must be positive")
    torch.set_num_threads(1)
    runs = [run_arm(seed, arm, episodes, eval_trials) for seed in seeds for arm in arms]
    aggregate = {}
    for train_arm in arms:
        group = [r for r in runs if r["train_arm"] == train_arm]
        aggregate[train_arm] = {
            "training_success_mean": float(np.mean([r["train_success_rate"] for r in group])),
            "matched_environment_test_success_mean": float(np.mean([
                r["end_eval"][train_arm]["success_rate"] for r in group
            ])),
            "heldout_cross_condition_success_mean": float(np.mean([
                r["end_eval"][test_arm]["success_rate"]
                for r in group for test_arm in ARMS if test_arm != train_arm
            ])),
            "heldout_cross_condition_change_from_untrained": float(np.mean([
                r["end_eval"][test_arm]["success_rate"] - r["start_eval"][test_arm]["success_rate"]
                for r in group for test_arm in ARMS if test_arm != train_arm
            ])),
        }
    return {
        "protocol": "space_time_energy_factorial_v1",
        "interpretation": (
            "Training environments vary the named limits. All agents use the "
            "same seeded recurrent actor-critic architecture and learning rule. "
            "Each agent is tested without gradient updates on independent "
            "matched-seed start/goal pairs in all eight evaluation conditions. "
            "Low episode counts are an engineering smoke test, not evidence "
            "of general intelligence or reliable cross-condition transfer."
        ),
        "seeds": seeds,
        "training_arms": arms,
        "eval_arms": list(ARMS),
        "episodes_per_arm": episodes,
        "heldout_trials_per_eval_arm": eval_trials,
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
        "settings": {arm: asdict(arm_config(arm)) for arm in ARMS},
        "runs": runs,
        "summary": aggregate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="CAESAR constraint-factorial and transfer test")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 47])
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--eval-trials", type=int, default=30)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--output", default="outputs/constraint_transfer.json")
    args = parser.parse_args()
    report = run_study(args.seeds, args.episodes, args.eval_trials, args.arms)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "protocol": report["protocol"],
        "seeds": report["seeds"],
        "training_arms": report["training_arms"],
        "episodes_per_arm": report["episodes_per_arm"],
        "eval_trials": report["heldout_trials_per_eval_arm"],
        "summary": report["summary"],
        "limit": "No significance or generalization claim from a smoke test",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
