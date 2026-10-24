"""Diagnostic: does the stock CAESAR scar trigger in real arena episodes?

No training updates. Records actual forward-call eligibility and scar increments.
This is a smoke test, NOT a controlled training result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .core import CAESARNet, EmbodiedEnv
from .training import set_seed


def probe(seed: int = 42, episodes: int = 3, level: int = 0) -> dict:
    if episodes < 1 or level < 0:
        raise ValueError("episodes must be positive and level nonnegative")
    set_seed(seed)
    torch.set_num_threads(1)
    model = CAESARNet().eval()
    env = EmbodiedEnv(level=level, rng=np.random.default_rng(seed))
    eligible_steps = 0
    scar_changes = 0
    observed_steps = 0
    lowest_battery = 100.0
    lowest_mirror = 1.0
    successes = 0
    epsilon = model.config.scar_increment * 0.5
    with torch.no_grad():
        for _ in range(episodes):
            observation = env.reset()
            hidden = torch.zeros(1, 1, model.hidden_size)
            done = False
            while not done:
                before_scar = float(model.scar.item())
                before_battery = float(env.battery)
                inp = torch.as_tensor(observation, dtype=torch.float32).view(1, 1, -1)
                mean, _, hidden = model(inp, hidden, before_battery)
                mirror = float(model.M_state.item())
                eligible_steps += int(
                    mirror < model.config.scar_threshold
                    and before_battery < model.config.scar_battery_threshold
                )
                scar_changes += int(float(model.scar.item()) - before_scar > epsilon)
                lowest_battery = min(lowest_battery, before_battery)
                lowest_mirror = min(lowest_mirror, mirror)
                # Deterministic untrained actions, to separate trigger reachability
                # from actor-critic learning.
                observation, _, done, info = env.step(mean.squeeze(0).squeeze(0).numpy())
                observed_steps += 1
            successes += int(info["success"])
    return {
        "seed": seed,
        "episodes": episodes,
        "level": level,
        "steps": observed_steps,
        "successes": successes,
        "scar_eligible_steps": eligible_steps,
        "scar_increment_events": scar_changes,
        "final_scar": float(model.scar.item()),
        "min_battery_at_forward": lowest_battery,
        "min_mirror": lowest_mirror,
        "scar_threshold": model.config.scar_threshold,
        "scar_battery_threshold": model.config.scar_battery_threshold,
        "interpretation": (
            "If scar_increment_events == 0, this setup does not exercise the "
            "scar mechanism. Do not interpret paired learning results as evidence "
            "about scar unless activation is independently confirmed."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="CAESAR forward scar-trigger diagnostic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--scan", action="store_true", help="Run a preselected 5-seed, 2-level stock-environment scan")
    parser.add_argument("--output", default="outputs/scar_trigger_probe.json")
    args = parser.parse_args()
    if args.scan:
        seeds = [11, 23, 47, 89, 101]
        levels = [0, 10]
        rows = [probe(seed=seed, episodes=args.episodes, level=level)
                for seed in seeds for level in levels]
        result = {
            "design": "fixed-seed forward-only stock-env scar activation reachability",
            "results": rows,
            "runs": len(rows),
            "runs_with_scar_activation": sum(row["scar_increment_events"] > 0 for row in rows),
            "limitations": "Does not train the policy or measure scar memory's effect on learning",
        }
    else:
        result = probe(args.seed, args.episodes, args.level)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
