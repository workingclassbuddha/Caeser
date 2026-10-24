"""Reproduce corrected CAESAR curriculum dynamics with explicit seeded runs.

The three archived console logs predate seed/environment-loss fixes, and did
not preserve enough settings for exact replay. This is a prospective
reproduction, not a claim to reproduce those exact historical trajectories.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .training import TrainingConfig, train_caesar


def replay(seeds: list[int], episodes: int, output: str) -> dict:
    if not seeds or len(seeds) != len(set(seeds)) or episodes < 1:
        raise ValueError("Need unique seeds and positive episodes")
    torch.set_num_threads(1)
    experiments = []
    for seed in seeds:
        path = Path(output).with_name(f"curriculum_seed_{seed}.json")
        history = train_caesar(TrainingConfig(
            episodes=episodes,
            seed=seed,
            device="cpu",
            log_every=20,
            output_path=str(path),
        ))
        records = [asdict(point) for point in history]
        experiments.append({
            "seed": seed,
            "episode_count": episodes,
            "last_logged_level": records[-1]["level"] if records else None,
            "last_logged_success_rate": records[-1]["success_rate"] if records else None,
            "max_logged_scar": max((record["scar"] for record in records), default=None),
            "snapshots": records,
            "snapshot_path": str(path),
        })
    report = {
        "protocol": "corrected_caesar_longrun_curriculum_v1",
        "caution": (
            "Prospective seeded study using fixed environment and corrected "
            "learning code. Archived 2026-April logs do not have recoverable "
            "random seeds/configuration, so numerical identity is not expected. "
            "Level promotions are explicit thresholds, not discovered events."
        ),
        "seeds": seeds,
        "episodes": episodes,
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
        "runs": experiments,
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Seeded corrected CAESAR curriculum replay")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 47])
    parser.add_argument("--episodes", type=int, default=7000)
    parser.add_argument("--output", default="outputs/curriculum_replication.json")
    args = parser.parse_args()
    result = replay(args.seeds, args.episodes, args.output)
    print(json.dumps({
        "protocol": result["protocol"], "seeds": args.seeds,
        "episodes": args.episodes, "summary": [
            {"seed": r["seed"], "last_logged_level": r["last_logged_level"],
             "max_logged_scar": r["max_logged_scar"]}
            for r in result["runs"]
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
