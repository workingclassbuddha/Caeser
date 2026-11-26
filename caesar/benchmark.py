"""Paired-seed CAESAR scar ablation. Research harness, not evidence by itself."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .experiments import ExperimentCondition, NoScarCondition, train_condition


def metrics(log: list[dict[str, float | int]]) -> dict[str, float | int | None]:
    """Extract outcome measures; mirror is a diagnostic, not the primary endpoint."""
    if not log:
        return {
            "last_success_rate": None,
            "mean_success_rate": None,
            "final_level": None,
            "max_scar": None,
            "first_level_gain_episode": None,
        }
    return {
        "last_success_rate": float(log[-1]["success_rate"]),
        "mean_success_rate": float(np.mean([float(x["success_rate"]) for x in log])),
        "final_level": int(log[-1]["level"]),
        "max_scar": float(max(float(x["scar"]) for x in log)),
        "first_level_gain_episode": next(
            (int(x["episode"]) for x in log if int(x["level"]) > 0), None
        ),
    }


def paired_summary(pairs: list[dict]) -> dict:
    """Report observed paired differences without declaring statistical significance."""
    if not pairs:
        raise ValueError("At least one pair is required")
    score_keys = ("last_success_rate", "mean_success_rate", "final_level")
    deltas = {
        key: [
            float(p["baseline"][key]) - float(p["no_scar"][key])
            for p in pairs
        ]
        for key in score_keys
    }
    return {
        "paired_seeds": len(pairs),
        "difference_definition": "baseline minus no_scar",
        "paired_differences": deltas,
        "mean_paired_differences": {k: float(np.mean(v)) for k, v in deltas.items()},
        "baseline_any_scar": any(float(p["baseline"]["max_scar"]) > 0 for p in pairs),
        "interpretation_limit": (
            "Scar ablation is uninformative if baseline scar never accumulates. "
            "Mirror is not a validated cognitive or consciousness measure."
        ),
    }


def run_benchmark(
    seeds: list[int],
    episodes: int,
    output: str,
    device: str = "cpu",
    log_every: int = 20,
) -> dict:
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Supply one or more unique seeds")
    if episodes <= 0 or log_every <= 0:
        raise ValueError("episodes and log_every must be positive")

    pairs: list[dict] = []
    for seed in seeds:
        baseline = train_condition(
            ExperimentCondition("BASELINE", "Persistent scar enabled"),
            max_episodes=episodes, seed=seed, device_name=device, log_every=log_every,
        )
        no_scar = train_condition(
            NoScarCondition(),
            max_episodes=episodes, seed=seed, device_name=device, log_every=log_every,
        )
        pairs.append({
            "seed": seed,
            "baseline": metrics(baseline.log),
            "no_scar": metrics(no_scar.log),
            "logs": {"baseline": baseline.log, "no_scar": no_scar.log},
        })

    report = {
        "design": "matched initialization and seeded environment per condition",
        "episodes_per_condition": episodes,
        "log_every": log_every,
        "device": device,
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
        "pairs": pairs,
        "summary": paired_summary(pairs),
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paired CAESAR scar ablation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 47, 89, 101])
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--output", default="outputs/paired_scar_ablation.json")
    args = parser.parse_args()
    report = run_benchmark(args.seeds, args.episodes, args.output, args.device, args.log_every)
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
