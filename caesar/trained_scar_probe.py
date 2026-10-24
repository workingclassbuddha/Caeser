"""Train the stock CAESAR actor-critic and measure scar activation.

This is a bounded diagnostic, not a no-scar ablation or evidence of benefit.
Unlike scar_probe.py, actions are sampled and the policy is updated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .core import CAESARNet, EmbodiedEnv
from .training import (
    TrainingConfig, apply_actor_critic_update, compute_noise_scale,
    run_training_episode, set_seed,
)


def trained_probe(seed: int = 42, episodes: int = 5, level: int = 0) -> dict:
    if episodes < 1 or level < 0:
        raise ValueError("episodes must be positive; level nonnegative")

    set_seed(seed)
    torch.set_num_threads(1)
    config = TrainingConfig(seed=seed, device="cpu", episodes=episodes)
    env = EmbodiedEnv(level=level, rng=np.random.default_rng(seed))
    agent = CAESARNet().train()
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate)
    records = []
    total_steps = 0
    total_scar_events = 0

    for episode in range(episodes):
        scar_before = float(agent.scar.item())
        rollout = run_training_episode(
            env, agent, torch.device("cpu"), compute_noise_scale(episode, config)
        )
        # Scar accumulates during forward passes, before the policy update.
        scar_after = float(agent.scar.item())
        change = max(0.0, scar_after - scar_before)
        events = int(round(change / agent.config.scar_increment))
        total_scar_events += events
        policy_loss, value_loss = apply_actor_critic_update(
            agent, optimizer, rollout, config, torch.device("cpu")
        )
        total_steps += rollout.steps
        records.append({
            "episode": episode,
            "steps": rollout.steps,
            "success": rollout.success,
            "end_battery": float(env.battery),
            "end_mirror": float(agent.M_state.item()),
            "scar_increment_events_estimate": events,
            "scar_before": scar_before,
            "scar_after": scar_after,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
        })

    return {
        "design": "seeded, actual actor-critic updates under stock environment constraints",
        "seed": seed,
        "episodes": episodes,
        "level": level,
        "total_steps": total_steps,
        "total_scar_events_estimate": total_scar_events,
        "episodes_with_scar": sum(x["scar_increment_events_estimate"] > 0 for x in records),
        "final_scar": float(agent.scar.item()),
        "scar_threshold": agent.config.scar_threshold,
        "scar_battery_threshold": agent.config.scar_battery_threshold,
        "records": records,
        "limitations": (
            "Short trained-policy activation diagnostic only. It does not test "
            "whether scar improves learning, and zero events cannot rule out "
            "activation after much longer training or different environments."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="CAESAR trained-policy scar activation diagnostic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--output", default="outputs/trained_scar_probe.json")
    args = parser.parse_args()
    result = trained_probe(seed=args.seed, episodes=args.episodes, level=args.level)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "seed": result["seed"],
        "episodes": result["episodes"],
        "level": result["level"],
        "total_steps": result["total_steps"],
        "total_scar_events_estimate": result["total_scar_events_estimate"],
        "episodes_with_scar": result["episodes_with_scar"],
        "final_scar": result["final_scar"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
