"""Pair CAESAR's held-out trials with nonlearning controllers by identical seeds.

This reports outcome differences, NOT inferential significance. Independent
training seeds are the experimental replication units, not individual trials.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compare(transfer: dict, controls: dict) -> dict:
    if transfer.get("protocol") != "space_time_energy_factorial_v1":
        raise ValueError("Unrecognized learned-agent report")
    if controls.get("protocol") != "caesar_nonlearning_observation_controller_v1":
        raise ValueError("Unrecognized controller report")
    if transfer["seeds"] != controls["seeds"]:
        raise ValueError("Seed mismatches invalidate matched comparisons")
    if transfer["heldout_trials_per_eval_arm"] != controls["heldout_trials_per_test_arm"]:
        raise ValueError("Test trial counts must be identical")
    if transfer["eval_arms"] != controls["eval_arms"]:
        raise ValueError("Test arm order/settings must match")
    if len(transfer["runs"]) != len(transfer["seeds"]) * len(transfer["training_arms"]):
        raise ValueError("Expected one learned model for every training seed and arm")

    per_training_arm = {}
    for train_arm in transfer["training_arms"]:
        matching_runs = [run for run in transfer["runs"] if run["train_arm"] == train_arm]
        if [run["seed"] for run in matching_runs] != transfer["seeds"]:
            raise ValueError("Missing or reordered learned runs in training arm " + train_arm)
        results = {}
        for controller in controls["controllers"]:
            seed_summaries = []
            for run in matching_runs:
                seed = run["seed"]
                pairs = []
                for test_arm in transfer["eval_arms"]:
                    if test_arm == train_arm:
                        continue  # cross-condition held-out measure, as preregistered
                    learned = run["end_eval"][test_arm]["trial_success"]
                    control = controls["evaluations"][controller][str(seed)][test_arm]["trial_success"]
                    if len(learned) != len(control) or len(learned) != transfer["heldout_trials_per_eval_arm"]:
                        raise ValueError("Missing/misaligned trial outcomes")
                    pairs.extend(int(a) - int(b) for a, b in zip(learned, control))
                seed_summaries.append({
                    "seed": seed,
                    "paired_test_count": len(pairs),
                    "learned_minus_controller_success": float(np.mean(pairs)),
                })
            results[controller] = {
                "comparison": "learned CAESAR minus fixed nonlearning controller",
                "per_training_seed": seed_summaries,
                "mean_seed_paired_success_delta": float(np.mean([
                    row["learned_minus_controller_success"] for row in seed_summaries
                ])),
                "claim_limit": (
                    "Three training seeds and short training horizons are "
                    "insufficient for reliable inferential claims; repeated "
                    "test cases are dependent within trained-policy seed."
                ),
            }
        per_training_arm[train_arm] = results

    return {
        "protocol": "caesar_paired_no_learning_comparison_v1",
        "train_seeds": transfer["seeds"],
        "training_arms": transfer["training_arms"],
        "test_trials_per_arm_per_seed": transfer["heldout_trials_per_eval_arm"],
        "comparison": per_training_arm,
        "interpretation": (
            "Goal direction is directly present in CAESAR's observation. "
            "A controller that navigates well without learning challenges "
            "attribution of navigation success to neural policy learning. "
            "No hyperparameters were selected after viewing results."
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Compare CAESAR to fixed nonlearning controls")
    p.add_argument("--learned", required=True)
    p.add_argument("--controls", required=True)
    p.add_argument("--output", default="outputs/learned_vs_controllers.json")
    args = p.parse_args()
    learned = json.loads(Path(args.learned).read_text())
    controls = json.loads(Path(args.controls).read_text())
    result = compare(learned, controls)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "protocol": result["protocol"],
        "train_seeds": result["train_seeds"],
        "mean_seed_paired_cross_condition_deltas": {
            arm: {
                controller: metrics["mean_seed_paired_success_delta"]
                for controller, metrics in results.items()
            } for arm, results in result["comparison"].items()
        },
        "limit": "Descriptive paired data; not a proof of learned general intelligence",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
