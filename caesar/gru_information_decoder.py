"""Measure whether CAESAR's recurrent state *retains decodable gap information*.

A frozen linear decoder is fitted on disjoint paired-world seeds and evaluated
on unseen paired-world seeds. It never trains the CAESAR actor or modifies its
observations. Compare trained and untrained GRUs, and history preserved versus
history reset on exactly matched current observations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .core import CAESARNet
from .memory_causal_probe import paired_histories, replay_policy
from .training import (
    TrainingConfig, apply_actor_critic_update, compute_noise_scale,
    run_training_episode, set_seed,
)
from .challenge_env import BarrierEnv, challenge_config

FEATURES = ("early_observation", "current_observation", "gru_preserved", "gru_reset")
FIXED_RIDGE_ALPHA = 25.0
TRAIN_WORLD_SEEDS = (31, 37, 41)
TEST_WORLD_SEEDS = (89, 101, 211)


def feature_rows(agent: CAESARNet, world_seeds: tuple[int, ...], trials: int) -> list[dict]:
    """Use only observation-matched top/bottom pairs in the causal decoding set."""
    if trials < 1 or not world_seeds or len(set(world_seeds)) != len(world_seeds):
        raise ValueError("invalid world seeds/trials")
    examples = []
    for world_seed in world_seeds:
        for trial in range(trials):
            paired = paired_histories(world_seed, trial)
            if not paired["same_current_obs"]:
                continue
            for side in ("top", "bottom"):
                history = paired["histories"][side]
                _, preserved = replay_policy(agent, history, reset_hidden=False)
                _, reset = replay_policy(agent, history, reset_hidden=True)
                examples.append({
                    "world_seed": world_seed, "trial": trial, "label": side,
                    "target": 1 if side == "top" else -1,
                    "approach_steps": paired["steps"],
                    "early_observation": [float(x) for x in history[0]],
                    "current_observation": [float(x) for x in history[-1]],
                    "gru_preserved": [float(x) for x in preserved],
                    "gru_reset": [float(x) for x in reset],
                })
    if not examples:
        raise RuntimeError("no observation-matched pairs")
    return examples


def ridge_probe(train: list[dict], test: list[dict], feature: str, *, shuffle_labels: bool = False) -> dict:
    """Fixed-regularization linear readout. No tuning on held-out worlds."""
    if feature not in FEATURES:
        raise ValueError("unrecognized feature")
    if not train or not test:
        raise ValueError("empty decode set")
    train_key = {(r["world_seed"], r["trial"]) for r in train}
    test_key = {(r["world_seed"], r["trial"]) for r in test}
    if train_key.intersection(test_key) or {
        r["world_seed"] for r in train
    }.intersection({r["world_seed"] for r in test}):
        raise ValueError("decoder train and test world seeds overlap")
    x = np.asarray([row[feature] for row in train], dtype=np.float64)
    z = np.asarray([row[feature] for row in test], dtype=np.float64)
    y = np.asarray([row["target"] for row in train], dtype=np.float64)
    actual = np.asarray([row["target"] for row in test], dtype=np.float64)
    if shuffle_labels:
        y = np.random.default_rng(4319).permutation(y)
    mu = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    xn = (x - mu) / scale
    zn = (z - mu) / scale
    ymean = y.mean()
    yn = y - ymean
    weights = np.linalg.solve(
        xn.T @ xn + FIXED_RIDGE_ALPHA * np.eye(xn.shape[1]), xn.T @ yn
    )
    scores = zn @ weights + ymean
    predictions = np.where(scores >= 0, 1, -1)
    return {
        "feature": feature,
        "shuffle_training_labels": shuffle_labels,
        "alpha": FIXED_RIDGE_ALPHA,
        "train_examples": len(train),
        "test_examples": len(test),
        "test_accuracy": float(np.mean(predictions == actual)),
        "test_top_accuracy": float(np.mean(predictions[actual == 1] == 1)),
        "test_bottom_accuracy": float(np.mean(predictions[actual == -1] == -1)),
        "test_predictions": [
            {
                "world_seed": row["world_seed"],
                "trial": row["trial"], "true_label": row["label"],
                "predicted_label": "top" if pred == 1 else "bottom",
                "signed_score": float(score),
            }
            for row, pred, score in zip(test, predictions, scores)
        ],
    }


def one_bit_accuracy(rows: list[dict]) -> dict:
    """Fixed memory comparator: initial NE ray versus initial SE ray."""
    guesses = [
        "top" if row["early_observation"][7] >= row["early_observation"][13]
        else "bottom"
        for row in rows
    ]
    return {
        "test_examples": len(rows),
        "accuracy": float(np.mean([
            guess == row["label"] for guess, row in zip(guesses, rows)
        ])),
    }


def probe_one_model(seed: int, episodes: int, train_trials: int, test_trials: int) -> dict:
    if episodes < 0:
        raise ValueError("episodes must be nonnegative")
    set_seed(seed)
    agent = CAESARNet()
    records = []
    stages = {}
    for stage in ("untrained", "trained"):
        if stage == "trained":
            if episodes:
                env = BarrierEnv(config=challenge_config("none"), rng=np.random.default_rng(seed))
                cfg = TrainingConfig(seed=seed, episodes=episodes, device="cpu")
                optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate)
                for ep in range(episodes):
                    rollout = run_training_episode(
                        env, agent, torch.device("cpu"), compute_noise_scale(ep, cfg)
                    )
                    apply_actor_critic_update(
                        agent, optimizer, rollout, cfg, torch.device("cpu")
                    )
                    records.append({
                        "episode": ep, "success": int(rollout.success),
                        "steps": rollout.steps, "return": float(rollout.total_reward),
                    })
        train = feature_rows(agent, TRAIN_WORLD_SEEDS, train_trials)
        test = feature_rows(agent, TEST_WORLD_SEEDS, test_trials)
        # Results rely on genuinely held-out *world seeds*, not just trials.
        readings = {
            name: ridge_probe(train, test, name)
            for name in FEATURES
        }
        shuffled = ridge_probe(train, test, "gru_preserved", shuffle_labels=True)
        stages[stage] = {
            "training_episodes_completed": 0 if stage == "untrained" else episodes,
            "train_pairs": len(train) // 2, "test_pairs": len(test) // 2,
            "probe_results": readings,
            "shuffled_label_control": shuffled,
            "fixed_one_bit_test": one_bit_accuracy(test),
            "train_features": train,
            "test_features": test,
        }
    return {
        "policy_seed": seed,
        "training_episodes": episodes,
        "training_successes": int(sum(row["success"] for row in records)),
        "training_trace": records,
        "stages": stages,
    }


def run_audit(
    policy_seeds: list[int], episodes: int, train_trials: int, test_trials: int
) -> dict:
    if (
        not policy_seeds or len(set(policy_seeds)) != len(policy_seeds) or
        episodes < 0 or train_trials < 1 or test_trials < 1
    ):
        raise ValueError("invalid policy seeds or lengths")
    if set(TRAIN_WORLD_SEEDS).intersection(TEST_WORLD_SEEDS):
        raise AssertionError("decoder world seeds overlap")
    torch.set_num_threads(1)
    policies = [
        probe_one_model(seed, episodes, train_trials, test_trials)
        for seed in policy_seeds
    ]
    compact = {
        stage: {
            feature: {
                "per_policy_seed_accuracy": {
                    str(p["policy_seed"]): p["stages"][stage]["probe_results"][feature]["test_accuracy"]
                    for p in policies
                },
                "mean_policy_seed_accuracy": float(np.mean([
                    p["stages"][stage]["probe_results"][feature]["test_accuracy"]
                    for p in policies
                ])),
            }
            for feature in FEATURES
        }
        for stage in ("untrained", "trained")
    }
    return {
        "protocol": "caesar_v2_gru_gap_information_linear_readout_v1",
        "policy_seeds": policy_seeds,
        "model_train_episodes": episodes,
        "decoder_train_world_seeds": list(TRAIN_WORLD_SEEDS),
        "decoder_test_world_seeds": list(TEST_WORLD_SEEDS),
        "decoder_train_trials_per_seed": train_trials,
        "decoder_test_trials_per_seed": test_trials,
        "fixed_ridge_alpha": FIXED_RIDGE_ALPHA,
        "classes": ["top", "bottom"],
        "method": (
            "Matched top/bottom world pairs are retained only if the final "
            "14-sensor vector is identical to 1e-6. Frozen linear ridge "
            "classifier fits on disjoint world seeds then tests elsewhere. "
            "Compare untrained and normally actor-critic trained CAESAR. "
            "A high decoder score is information present, not learned "
            "use by the ACTOR. Final-observation-only and last-step GRU-reset "
            "controls quantify immediate sensory leakage, and a shuffled "
            "training-label probe checks spurious decoding."
        ),
        "policies": policies, "summary": compact,
        "limitations": (
            "Decoder is externally trained with labeled counterfactual "
            "gap sides, NOT CAESAR. Learned hidden representations may "
            "also occur in randomly initialized GRUs. This is not "
            "end-to-end navigation success, conscious memory, or a "
            "new general theory of intelligence."
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Frozen linear-probe GRU memory information audit")
    p.add_argument("--policy-seeds", nargs="+", type=int, default=[11, 23])
    p.add_argument("--train-episodes", type=int, default=50)
    p.add_argument("--train-trials", type=int, default=24)
    p.add_argument("--test-trials", type=int, default=24)
    p.add_argument("--output", default="outputs/gru_information_decoder.json")
    args = p.parse_args()
    report = run_audit(
        args.policy_seeds, args.train_episodes, args.train_trials, args.test_trials
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "protocol": report["protocol"],
        "policy_seeds": report["policy_seeds"],
        "model_train_episodes": report["model_train_episodes"],
        "summary": report["summary"],
        "per_policy": [
            {
                "seed": p["policy_seed"], "training_successes": p["training_successes"],
                "decoders": {
                    stage: {
                        "train_pairs": p["stages"][stage]["train_pairs"],
                        "test_pairs": p["stages"][stage]["test_pairs"],
                        "test_accuracy": {
                            feat: p["stages"][stage]["probe_results"][feat]["test_accuracy"]
                            for feat in FEATURES
                        },
                        "shuffled_training_labels_accuracy": p["stages"][stage][
                            "shuffled_label_control"
                        ]["test_accuracy"],
                        "one_bit_test_accuracy": p["stages"][stage]["fixed_one_bit_test"]["accuracy"],
                    }
                    for stage in ("untrained", "trained")
                },
            }
            for p in report["policies"]
        ],
        "caution": report["limitations"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
