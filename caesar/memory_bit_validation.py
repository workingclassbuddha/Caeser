"""Independent validation of the precommitted one-bit memory controller.

No tuning of cue, thresholds, crossing interval or retry logic on new seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .challenge_env import CHALLENGE_ARMS
from .memory_causal_probe import CONTROLLERS, evaluate_scanner, paired_histories


def validate(seeds: list[int], trials: int, pairs_per_seed: int) -> dict:
    if not seeds or len(set(seeds)) != len(seeds) or min(trials, pairs_per_seed) < 1:
        raise ValueError("distinct seeds and positive counts required")
    rows = {
        str(seed): {
            arm: {
                control: evaluate_scanner(seed, trials, arm, control)
                for control in CONTROLLERS
            }
            for arm in CHALLENGE_ARMS
        } for seed in seeds
    }
    summary = {}
    for control in CONTROLLERS:
        outcome = [
            trial for seed in seeds for arm in CHALLENGE_ARMS
            for trial in rows[str(seed)][arm][control]["trial_log"]
        ]
        summary[control] = {
            "n": len(outcome), "successes": int(sum(r["success"] for r in outcome)),
            "success_rate": float(np.mean([r["success"] for r in outcome])),
            "tight_time_success": float(np.mean([
                r["success"] for seed in seeds for arm in CHALLENGE_ARMS
                if "time" in arm
                for r in rows[str(seed)][arm][control]["trial_log"]
            ])),
            "ordinary_time_success": float(np.mean([
                r["success"] for seed in seeds for arm in CHALLENGE_ARMS
                if "time" not in arm
                for r in rows[str(seed)][arm][control]["trial_log"]
            ])),
            "top_success": float(np.mean([
                r["success"] for r in outcome if r["actual_gap"] == "top"
            ])),
            "bottom_success": float(np.mean([
                r["success"] for r in outcome if r["actual_gap"] == "bottom"
            ])),
        }
    cue = []
    for seed in seeds:
        for trial in range(pairs_per_seed):
            p = paired_histories(seed, trial)
            top = p["histories"]["top"][0]
            bottom = p["histories"]["bottom"][0]
            cue.append({
                "seed": seed, "trial": trial,
                "top_guess": "top" if top[7] >= top[13] else "bottom",
                "bottom_guess": "top" if bottom[7] >= bottom[13] else "bottom",
                "top_correct": bool(top[7] >= top[13]),
                "bottom_correct": bool(bottom[7] < bottom[13]),
            })
    return {
        "protocol": "caesar_v2_frozen_one_bit_memory_independent_validation_v1",
        "seeds": seeds, "trials_per_arm_per_seed": trials,
        "pairs_per_seed": pairs_per_seed,
        "controls": list(CONTROLLERS),
        "frozen_rule": (
            "Store bit at start: ray northeast>=southeast implies top. "
            "At wall, scan stored side first using same fixed finite-state "
            "wall-following controller. No weights, oracle gap label, "
            "threshold tuning or neural changes."
        ),
        "summary": summary,
        "cue_accuracy": {
            "top": float(np.mean([c["top_correct"] for c in cue])),
            "bottom": float(np.mean([c["bottom_correct"] for c in cue])),
            "overall": float(np.mean([
                (c["top_correct"] + c["bottom_correct"]) / 2 for c in cue
            ])),
        },
        "cue_trials": cue,
        "evaluations": rows,
        "limits": (
            "This is a hand-written memory strategy in a fixed two-opening "
            "task. Strong baseline success means neural learning must beat "
            "or match this strategy before claims of learned adaptive use. "
            "Different obstacle topology and noisy sensors still untested."
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Freeze-then-validate one-bit gap memory on new seeds")
    p.add_argument("--seeds", nargs="+", type=int, default=[89, 101, 211])
    p.add_argument("--trials", type=int, default=16)
    p.add_argument("--pairs", type=int, default=30)
    p.add_argument("--output", default="outputs/one_bit_independent_validation.json")
    args = p.parse_args()
    report = validate(args.seeds, args.trials, args.pairs)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "protocol": report["protocol"], "seeds": report["seeds"],
        "trials_per_arm_per_seed": report["trials_per_arm_per_seed"],
        "summary": report["summary"], "cue_accuracy": report["cue_accuracy"],
        "limits": report["limits"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
