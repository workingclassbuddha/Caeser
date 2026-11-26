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

from .core import CAESARNet, EmbodiedEnv, EnvConfig
from .training import (
    TrainingConfig, apply_actor_critic_update, compute_noise_scale,
    run_training_episode, set_seed,
)


def trained_probe(
    seed: int = 42, episodes: int = 5, level: int = 0,
    max_steps: int | None = None,
) -> dict:
    if episodes < 1 or level < 0 or (max_steps is not None and max_steps < 1):
        raise ValueError("episodes/max_steps must be positive; level nonnegative")

    set_seed(seed)
    torch.set_num_threads(1)
    config = TrainingConfig(seed=seed, device="cpu", episodes=episodes)
    env_config = EnvConfig(base_max_steps=max_steps) if max_steps is not None else None
    env = EmbodiedEnv(level=level, config=env_config, rng=np.random.default_rng(seed))
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
        "base_max_steps": env.config.base_max_steps,
        "horizon_override": max_steps is not None,
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
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Exploratory stress arm: override episode horizon; all other dynamics unchanged")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Preselect matched training seeds; one run per seed")
    parser.add_argument("--output", default="outputs/trained_scar_probe.json")
    args = parser.parse_args()
    if args.seeds is not None:
        if not args.seeds or len(set(args.seeds)) != len(args.seeds):
            parser.error("--seeds requires distinct seed values")
        runs = [trained_probe(seed=value, episodes=args.episodes, level=args.level, max_steps=args.max_steps)
                for value in args.seeds]
        result = {
            "design": "repeated independent seeded actor-critic training activation probes",
            "seeds": args.seeds,
            "episodes_per_seed": args.episodes,
            "level": args.level,
            "horizon_override": args.max_steps,
            "runs": runs,
            "runs_with_scar_events": sum(r["total_scar_events_estimate"] > 0 for r in runs),
            "total_training_steps": sum(r["total_steps"] for r in runs),
            "total_scar_events_estimate": sum(r["total_scar_events_estimate"] for r in runs),
            "limit": "No paired outcome comparison or conclusion beyond observed training reachability.",
        }
    else:
        result = trained_probe(
            seed=args.seed, episodes=args.episodes, level=args.level,
            max_steps=args.max_steps
        )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    if args.seeds is not None:
        print(json.dumps({
            "seeds": result["seeds"],
            "episodes_per_seed": result["episodes_per_seed"],
            "total_training_steps": result["total_training_steps"],
            "runs_with_scar_events": result["runs_with_scar_events"],
            "total_scar_events_estimate": result["total_scar_events_estimate"],
        }, indent=2))
    else:
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
