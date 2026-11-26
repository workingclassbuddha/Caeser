"""CAESAR v2 challenge: fixed nonlearning controls, reachability oracle, trained RL.

The reference planner is deliberately *privileged* and is only used as a
positive feasibility control. Direct controllers see exactly CAESAR's
observation; the learning agent uses unchanged training code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .baselines import controller_action
from .reactive_control import WallScanner, StoredRayBitScanner
from .challenge_env import BarrierEnv, CHALLENGE_ARMS, challenge_config
from .core import CAESARNet
from .training import (
    TrainingConfig, apply_actor_critic_update, compute_noise_scale,
    run_training_episode, set_seed,
)

CONTROLS = ("direct", "velocity_compensated", "stationary", "wall_scanner", "stored_ray_bit", "oracle_route")
EVAL_SEED_OFFSET = 9_000_000


def oracle_action(env: BarrierEnv, stage: int) -> tuple[np.ndarray, int]:
    """Privileged geometric route only for task-feasibility certification."""
    before_gap = np.array([env.wall_x - 0.32, env.gap_center], dtype=np.float32)
    after_gap = np.array([env.wall_x + 0.35, env.gap_center], dtype=np.float32)
    if stage == 0 and np.linalg.norm(env.agent_pos - before_gap) < 0.22:
        stage = 1
    if stage == 1 and env.agent_pos[0] > env.wall_x + 0.30:
        stage = 2
    target = before_gap if stage == 0 else after_gap if stage == 1 else env.goal_pos
    velocity = env.velocity
    # PD waypoint tracking, frozen as a feasibility controller, not trained.
    action = np.clip(3.0 * (target - env.agent_pos) - 4.0 * velocity, -1, 1)
    return action.astype(np.float32), stage


def evaluate(
    seed: int, arm: str, trials: int,
    controller: str | None = "direct",
    agent: CAESARNet | None = None,
) -> dict:
    if trials < 1:
        raise ValueError("trials must be positive")
    if (controller is None) == (agent is None):
        raise ValueError("supply exactly one of controller or agent")
    if controller is not None and controller not in CONTROLS:
        raise ValueError("unknown control")
    saved = None
    if agent is not None:
        saved = (agent.scar.clone(), agent.M_state.clone(), agent.training)
        agent.eval()
    outcomes = []
    try:
        with torch.no_grad():
            for trial in range(trials):
                rng = np.random.default_rng(EVAL_SEED_OFFSET + seed * 10_000 + trial)
                env = BarrierEnv(config=challenge_config(arm), rng=rng)
                observation = env.reset()
                if agent is not None:
                    assert saved is not None
                    agent.scar.copy_(saved[0])
                    agent.M_state.copy_(saved[1])
                hidden = torch.zeros(1, 1, agent.hidden_size) if agent is not None else None
                stage = 0
                scanner = (WallScanner() if controller == "wall_scanner" else
                           StoredRayBitScanner() if controller == "stored_ray_bit" else None)
                done = False
                total_reward = 0.0
                while not done:
                    if agent is not None:
                        tensor = torch.as_tensor(observation, dtype=torch.float32).view(1, 1, -1)
                        out, _, hidden = agent(tensor, hidden, env.battery)
                        action = out.reshape(-1).cpu().numpy()
                    elif controller == "oracle_route":
                        action, stage = oracle_action(env, stage)
                    elif controller in ("wall_scanner", "stored_ray_bit"):
                        assert scanner is not None
                        action = scanner.action(observation)
                    else:
                        assert controller is not None
                        action = controller_action(observation, controller)
                    observation, reward, done, info = env.step(action)
                    total_reward += reward
                outcomes.append({
                    "trial": trial,
                    "gap_side": env.gap_side,
                    "success": int(info["success"]),
                    "steps": env.steps,
                    "battery_left": float(env.battery),
                    "collisions": int(env.collision_count),
                    "return": float(total_reward),
                })
    finally:
        if saved is not None and agent is not None:
            with torch.no_grad():
                agent.scar.copy_(saved[0])
                agent.M_state.copy_(saved[1])
            agent.train(saved[2])
    return {
        "seed": seed, "arm": arm, "controller": controller or "caesar",
        "trials": trials, "successes": sum(x["success"] for x in outcomes),
        "success_rate": float(np.mean([x["success"] for x in outcomes])),
        "mean_steps": float(np.mean([x["steps"] for x in outcomes])),
        "mean_terminal_battery": float(np.mean([x["battery_left"] for x in outcomes])),
        "mean_collisions": float(np.mean([x["collisions"] for x in outcomes])),
        "trials_log": outcomes,
    }


def train_one(seed: int, arm: str, episodes: int, trials: int) -> dict:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    set_seed(seed)
    agent = CAESARNet()
    env = BarrierEnv(config=challenge_config(arm), rng=np.random.default_rng(seed))
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    config = TrainingConfig(episodes=episodes, device="cpu", seed=seed)
    before = {test_arm: evaluate(seed, test_arm, trials, controller=None, agent=agent)
              for test_arm in CHALLENGE_ARMS}
    episode_log = []
    for ep in range(episodes):
        rollout = run_training_episode(
            env, agent, torch.device("cpu"), compute_noise_scale(ep, config)
        )
        policy_loss, value_loss = apply_actor_critic_update(
            agent, optimizer, rollout, config, torch.device("cpu")
        )
        episode_log.append({
            "episode": ep, "success": int(rollout.success),
            "steps": rollout.steps, "return": rollout.total_reward,
            "mirror": float(agent.M_state.item()),
            "scar": float(agent.scar.item()),
            "policy_loss": policy_loss, "value_loss": value_loss,
        })
    after = {test_arm: evaluate(seed, test_arm, trials, controller=None, agent=agent)
             for test_arm in CHALLENGE_ARMS}
    return {
        "seed": seed, "train_arm": arm, "episodes": episodes,
        "train_success": float(np.mean([r["success"] for r in episode_log])),
        "pretrain": before, "posttrain": after, "train_log": episode_log,
    }


def run_challenge(
    seeds: list[int], episodes: int, trials: int, train_arms: list[str],
) -> dict:
    if not seeds or len(seeds) != len(set(seeds)) or not train_arms or len(set(train_arms)) != len(train_arms):
        raise ValueError("seeds and arms must be distinct, nonempty")
    if episodes < 1 or trials < 1 or any(x not in CHALLENGE_ARMS for x in train_arms):
        raise ValueError("invalid training length, trial count or arm")
    torch.set_num_threads(1)
    controls = {
        str(seed): {
            arm: {control: evaluate(seed, arm, trials, controller=control)
                  for control in CONTROLS}
            for arm in CHALLENGE_ARMS
        } for seed in seeds
    }
    training = [train_one(seed, arm, episodes, trials)
                for seed in seeds for arm in train_arms]
    controller_summary = {
        ctrl: {
            arm: float(np.mean([controls[str(seed)][arm][ctrl]["success_rate"] for seed in seeds]))
            for arm in CHALLENGE_ARMS
        } for ctrl in CONTROLS
    }
    trained_summary = {
        arm: {
            "trained_test_success": float(np.mean([
                run["posttrain"][arm]["success_rate"] for run in training if run["train_arm"] == arm
            ])),
            "pretrain_test_success": float(np.mean([
                run["pretrain"][arm]["success_rate"] for run in training if run["train_arm"] == arm
            ])),
            "training_episode_success": float(np.mean([
                run["train_success"] for run in training if run["train_arm"] == arm
            ])),
        } for arm in train_arms
    }
    return {
        "protocol": "caesar_v2_barrier_constraints_v1",
        "seeds": seeds, "training_arms": train_arms, "test_arms": list(CHALLENGE_ARMS),
        "training_episodes_per_arm": episodes, "heldout_trials_per_arm": trials,
        "torch": torch.__version__, "numpy": np.__version__,
        "methods": (
            "Unmodified CAESAR GRU actor-critic and reward/loss mechanics. "
            "Wall collision, eight ray hits, symmetric randomized top/bottom "
            "gap, deterministic observation-only controls, and PRIVILEGED "
            "oracle planner solely to certify reachability. Agent never "
            "observes oracle routes or privileged gap labels. Frozen "
            "network tested without gradient updates on disjoint seed cases."
        ),
        "controllers": controls, "training": training,
        "controller_summary": controller_summary,
        "trained_summary": trained_summary,
        "limitations": (
            "Small CI study checks feasibility and baseline failure, not "
            "a claim that learned intelligence emerges. Oracle possesses "
            "world geometry unavailable to the learned policy. "
            "The sample lacks statistical power and long training."
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate v2 wall-gap challenge arena")
    p.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 47])
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--train-arms", nargs="+", choices=CHALLENGE_ARMS,
                   default=list(CHALLENGE_ARMS))
    p.add_argument("--output", default="outputs/challenge_v2.json")
    args = p.parse_args()
    result = run_challenge(args.seeds, args.episodes, args.trials, args.train_arms)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "seeds": result["seeds"],
        "training_episodes_per_arm": args.episodes,
        "heldout_trials_per_arm": args.trials,
        "controls": result["controller_summary"],
        "learned": result["trained_summary"],
        "caution": result["limitations"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
