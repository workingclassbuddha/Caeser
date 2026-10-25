"""Causal audit: does a *trained* CAESAR policy use history to choose a gap?

No new model layers or altered training rule. A forced, matched approach is
used only in a diagnostic intervention, not as learned behavioural evidence.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .baselines import controller_action
from .challenge_benchmark import EVAL_SEED_OFFSET
from .challenge_env import BarrierEnv, CHALLENGE_ARMS, challenge_config
from .core import CAESARNet
from .reactive_control import WallScanner
from .training import (
    TrainingConfig, apply_actor_critic_update, compute_noise_scale,
    run_training_episode, set_seed,
)

TEST_OFFSET = 44_000_000
IDENTICAL_OBS_TOL = 1e-6
DIRECTION_TOL = 0.02
CONTROLLERS = ("default_up", "stored_ray_bit")


@dataclass
class StoredRayBitScanner(WallScanner):
    """One bit captured at spawn; all later decisions use the same sensors.

    Fixed physics-derived cue: north-east ray (index 7) vs south-east
    ray (index 13). Never accesses hidden gap position or label.
    """
    preferred_side: str | None = None

    def action(self, obs: np.ndarray) -> np.ndarray:
        if self.preferred_side is None:
            self.preferred_side = "top" if float(obs[7]) >= float(obs[13]) else "bottom"
        if self.phase == "approach" and float(obs[6]) < 0.065:
            if self.preferred_side == "bottom":
                self.phase = "search_down"
        return super().action(obs)


def paired_histories(seed: int, trial: int, arm: str = "none") -> dict:
    worlds = {}
    for side in ("top", "bottom"):
        env = BarrierEnv(
            config=challenge_config(arm),
            rng=np.random.default_rng(TEST_OFFSET + seed * 10_000 + trial),
            fixed_gap_side=side,
        )
        obs = env.reset()
        worlds[side] = (env, [obs.copy()])
    top, top_obs = worlds["top"]
    bottom, bottom_obs = worlds["bottom"]
    if not np.array_equal(top.agent_pos, bottom.agent_pos) or not np.array_equal(top.goal_pos, bottom.goal_pos):
        raise AssertionError("top and bottom paired worlds have different initial geometry")
    while float(top_obs[-1][6]) >= 0.065 and top.steps < 100:
        action = controller_action(top_obs[-1], "direct")
        for env, history in worlds.values():
            new_obs, _, done, _ = env.step(action)
            history.append(new_obs.copy())
            if done:
                raise AssertionError("episode ended before reaching the decision boundary")
        if not np.allclose(top.agent_pos, bottom.agent_pos, atol=1e-6):
            raise AssertionError("forced approach changed physics across paired worlds")
    return {
        "histories": {side: history for side, (_, history) in worlds.items()},
        "steps": top.steps,
        "same_current_obs": bool(
            np.max(np.abs(top_obs[-1] - bottom_obs[-1])) < IDENTICAL_OBS_TOL
        ),
        "max_current_obs_difference": float(np.max(np.abs(top_obs[-1] - bottom_obs[-1]))),
    }


def replay_policy(agent: CAESARNet, observations: list[np.ndarray], *, reset_hidden: bool) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate same final input with or without GRU history.

    Freeze model weights, mirror and scar buffers; the model's mirror state
    normally updates even in eval(), so save/restore it after every replay.
    """
    before_mode = agent.training
    before_m = agent.M_state.clone()
    before_s = agent.scar.clone()
    agent.eval()
    try:
        with torch.no_grad():
            h = torch.zeros(1, 1, agent.hidden_size)
            for obs in observations[:-1]:
                x = torch.from_numpy(obs).float().view(1, 1, -1)
                _, _, h = agent(x, h, float(obs[5]) * 50.0)
            if reset_hidden:
                h.zero_()
            last = observations[-1]
            x = torch.from_numpy(last).float().view(1, 1, -1)
            action, _, state = agent(x, h, float(last[5]) * 50.0)
            return action.reshape(-1).numpy().copy(), state.reshape(-1).numpy().copy()
    finally:
        with torch.no_grad():
            agent.M_state.copy_(before_m)
            agent.scar.copy_(before_s)
        agent.train(before_mode)


def probe_model(agent: CAESARNet, seed: int, trials: int) -> dict:
    cases = []
    for trial in range(trials):
        pair = paired_histories(seed, trial)
        histories = pair["histories"]
        preserved = {
            side: replay_policy(agent, hist, reset_hidden=False)
            for side, hist in histories.items()
        }
        reset = {
            side: replay_policy(agent, hist, reset_hidden=True)
            for side, hist in histories.items()
        }
        # History contrast after receiving precisely identical current input.
        same = pair["same_current_obs"]
        def measure(a):
            top, bottom = a["top"][0], a["bottom"][0]
            return {
                "top_y_action": float(top[1]),
                "bottom_y_action": float(bottom[1]),
                "action_pair_l2": float(np.linalg.norm(top - bottom)),
                "top_correct_sign": bool(float(top[1]) > DIRECTION_TOL),
                "bottom_correct_sign": bool(float(bottom[1]) < -DIRECTION_TOL),
                "both_correct": bool(
                    float(top[1]) > DIRECTION_TOL and float(bottom[1]) < -DIRECTION_TOL
                ),
            }
        cases.append({
            "trial": trial, "steps": pair["steps"],
            "same_current_obs": same,
            "max_current_obs_difference": pair["max_current_obs_difference"],
            "preserved": measure(preserved), "reset": measure(reset),
            "hidden_pair_l2": float(np.linalg.norm(preserved["top"][1] - preserved["bottom"][1])),
            "first_ray_cue": {
                side: {
                    "northeast": float(hist[0][7]),
                    "southeast": float(hist[0][13]),
                    "one_bit_side": "top" if hist[0][7] >= hist[0][13] else "bottom",
                } for side, hist in histories.items()
            },
        })
    matched = [r for r in cases if r["same_current_obs"]]
    def avg(rows, key):
        return float(np.mean([x[key] for x in rows])) if rows else None
    def metrics(mode):
        sub = [r[mode] for r in matched]
        return {
            "paired_worlds": len(sub),
            "mean_top_y_action": avg(sub, "top_y_action"),
            "mean_bottom_y_action": avg(sub, "bottom_y_action"),
            "mean_pair_action_l2": avg(sub, "action_pair_l2"),
            "fraction_correct_top_sign": avg(sub, "top_correct_sign"),
            "fraction_correct_bottom_sign": avg(sub, "bottom_correct_sign"),
            "fraction_both_signs_correct": avg(sub, "both_correct"),
        }
    return {
        "seed": seed, "pairs_total": trials,
        "matched_current_observation_pairs": len(matched),
        "preserved": metrics("preserved"),
        "hidden_reset": metrics("reset"),
        "memory_relevant_action_pair_fraction": float(np.mean([
            r["preserved"]["action_pair_l2"] > 1e-5 for r in matched
        ])) if matched else None,
        "one_bit_spawn_accuracy": float(np.mean([
            (r["first_ray_cue"]["top"]["one_bit_side"] == "top") * 0.5 +
            (r["first_ray_cue"]["bottom"]["one_bit_side"] == "bottom") * 0.5
            for r in cases
        ])),
        "pairs": cases,
    }


def evaluate_scanner(seed: int, trials: int, arm: str, policy: str) -> dict:
    if policy not in CONTROLLERS:
        raise ValueError("unknown scanner policy")
    outcomes = []
    for trial in range(trials):
        env = BarrierEnv(
            config=challenge_config(arm),
            rng=np.random.default_rng(EVAL_SEED_OFFSET + seed * 10_000 + trial),
        )
        obs = env.reset()
        controller = WallScanner() if policy == "default_up" else StoredRayBitScanner()
        done = False
        while not done:
            obs, _, done, info = env.step(controller.action(obs))
        outcomes.append({
            "trial": trial,
            "actual_gap": env.gap_side,
            "stored_guess": controller.preferred_side if isinstance(controller, StoredRayBitScanner) else None,
            "success": int(info["success"]),
            "steps": env.steps,
            "battery_remaining": float(env.battery),
        })
    return {
        "seed": seed, "arm": arm, "policy": policy, "trials": trials,
        "successes": sum(x["success"] for x in outcomes),
        "success_rate": float(np.mean([x["success"] for x in outcomes])),
        "trial_log": outcomes,
    }


def train_probe(seed: int, episodes: int, trials: int) -> dict:
    set_seed(seed)
    agent = CAESARNet()
    before = probe_model(agent, seed, trials)
    if episodes:
        env = BarrierEnv(
            config=challenge_config("none"), rng=np.random.default_rng(seed)
        )
        cfg = TrainingConfig(seed=seed, episodes=episodes, device="cpu")
        optim = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate)
        outcomes = []
        for ep in range(episodes):
            roll = run_training_episode(
                env, agent, torch.device("cpu"), compute_noise_scale(ep, cfg)
            )
            apply_actor_critic_update(agent, optim, roll, cfg, torch.device("cpu"))
            outcomes.append(int(roll.success))
    else:
        outcomes = []
    after = probe_model(agent, seed, trials)
    return {
        "seed": seed, "trained_episodes": episodes,
        "training_successes": sum(outcomes),
        "training_success_rate": float(np.mean(outcomes)) if outcomes else None,
        "untrained": before,
        "trained": after,
    }


def run_audit(seeds: list[int], train_episodes: int, probe_trials: int, control_trials: int) -> dict:
    if not seeds or len(seeds) != len(set(seeds)) or min(train_episodes, probe_trials, control_trials) < 0:
        raise ValueError("invalid seeds or counts")
    if probe_trials < 1 or control_trials < 1:
        raise ValueError("probe and control trials must be positive")
    torch.set_num_threads(1)
    trained = [train_probe(seed, train_episodes, probe_trials) for seed in seeds]
    controls = {
        str(seed): {
            arm: {
                policy: evaluate_scanner(seed, control_trials, arm, policy)
                for policy in CONTROLLERS
            } for arm in CHALLENGE_ARMS
        } for seed in seeds
    }
    summary = {}
    for name in CONTROLLERS:
        summary[name] = {
            arm: float(np.mean([
                controls[str(seed)][arm][name]["success_rate"] for seed in seeds
            ])) for arm in CHALLENGE_ARMS
        }
    return {
        "protocol": "caesar_v2_frozen_gru_history_reset_v1",
        "seeds": seeds, "trained_episodes": train_episodes,
        "probe_trials": probe_trials, "control_trials": control_trials,
        "method": (
            "Exactly matched top/bottom histories with forced shared goal "
            "approach. Only near-wall identical-observation pairs count "
            "toward paired causal action contrast. Frozen GRU parameters "
            "and same final observations; preserved history versus GRU reset. "
            "No behavior is attributed to learned navigation unless the "
            "directional sign tracks the true gap. One-bit controller uses "
            "only a fixed comparison of pre-existing northeast/southeast rays."
        ),
        "learners": trained, "controls": controls, "control_summary": summary,
        "limitations": (
            "Forced trajectories show capacity/use of GRU history at the "
            "decision point, not end-to-end learned pathfinding. The GRU "
            "may produce different actions from different histories even "
            "with random weights. Correct choices and transfer, not merely "
            "different hidden states, are the actual evidence gate."
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Causal recurrent history and one-bit control test")
    p.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 47])
    p.add_argument("--train-episodes", type=int, default=150)
    p.add_argument("--probe-trials", type=int, default=40)
    p.add_argument("--control-trials", type=int, default=30)
    p.add_argument("--output", default="outputs/gru_memory_causal_probe.json")
    args = p.parse_args()
    result = run_audit(args.seeds, args.train_episodes, args.probe_trials, args.control_trials)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "seeds": args.seeds,
        "training": [{
            "seed": r["seed"], "trained_episodes": r["trained_episodes"],
            "training_successes": r["training_successes"],
            "untrained": {
                "matched_pairs": r["untrained"]["matched_current_observation_pairs"],
                "preserved": r["untrained"]["preserved"],
                "reset": r["untrained"]["hidden_reset"],
                "one_bit_spawn_accuracy": r["untrained"]["one_bit_spawn_accuracy"],
            },
            "trained": {
                "matched_pairs": r["trained"]["matched_current_observation_pairs"],
                "preserved": r["trained"]["preserved"],
                "reset": r["trained"]["hidden_reset"],
                "one_bit_spawn_accuracy": r["trained"]["one_bit_spawn_accuracy"],
            },
        } for r in result["learners"]],
        "controls": result["control_summary"],
        "caution": result["limitations"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
