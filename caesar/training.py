from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from .core import CAESARNet, EmbodiedEnv, EnvConfig, ModelConfig


@dataclass(slots=True)
class TrainingConfig:
    """Top-level training settings."""

    episodes: int = 30_000
    learning_rate: float = 3e-4
    discount: float = 0.98
    value_loss_coef: float = 0.5
    gradient_clip_norm: float = 1.0
    success_window_size: int = 50
    log_every: int = 20
    initial_noise_scale: float = 1.2
    minimum_noise_scale: float = 0.25
    noise_decay_episodes: int = 1_200
    level_zero_promotion_threshold: float = 0.50
    standard_promotion_threshold: float = 0.70
    late_level_promotion_threshold: float = 0.60
    late_level_start: int = 10
    seed: int = 42
    device: str | None = None
    output_path: str | None = None


@dataclass(slots=True)
class EpisodeRollout:
    rewards: list[float]
    log_probs: list[torch.Tensor]
    values: list[torch.Tensor]
    success: bool
    total_reward: float
    steps: int


@dataclass(slots=True)
class TrainingSnapshot:
    episode: int
    level: int
    success_rate: float
    success_radius: float
    mirror: float
    scar: float
    total_reward: float
    steps: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested_device: str | None) -> torch.device:
    if requested_device:
        return torch.device(requested_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_noise_scale(episode_index: int, config: TrainingConfig) -> float:
    decay_fraction = episode_index / max(config.noise_decay_episodes, 1)
    return max(
        config.minimum_noise_scale,
        config.initial_noise_scale - decay_fraction,
    )


def promotion_threshold(level: int, config: TrainingConfig) -> float:
    if level == 0:
        return config.level_zero_promotion_threshold
    if level >= config.late_level_start:
        return config.late_level_promotion_threshold
    return config.standard_promotion_threshold


def discounted_returns(
    rewards: list[float],
    discount: float,
    device: torch.device,
) -> torch.Tensor:
    returns: list[float] = []
    accumulator = 0.0
    for reward in reversed(rewards):
        accumulator = reward + discount * accumulator
        returns.append(accumulator)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32, device=device)


def run_training_episode(
    env: EmbodiedEnv,
    agent: CAESARNet,
    device: torch.device,
    noise_scale: float,
) -> EpisodeRollout:
    observation = env.reset()
    hidden_state = torch.zeros(1, 1, agent.hidden_size, device=device)
    done = False

    rewards: list[float] = []
    log_probs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    total_reward = 0.0
    success = False

    while not done:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).view(1, 1, -1)
        action_mean, state_value, hidden_state = agent(obs_tensor, hidden_state, env.battery)

        action_std = torch.exp(agent.log_std).clamp(1e-3, 1.0) * noise_scale
        distribution = Normal(action_mean, action_std.view(1, 1, -1))
        action = distribution.sample()

        observation, reward, done, info = env.step(action.squeeze(0).squeeze(0).detach().cpu().numpy())
        rewards.append(reward)
        log_probs.append(distribution.log_prob(action).sum(dim=-1).reshape(()))
        values.append(state_value.reshape(()))
        total_reward += reward
        success = bool(info["success"])

    return EpisodeRollout(
        rewards=rewards,
        log_probs=log_probs,
        values=values,
        success=success,
        total_reward=total_reward,
        steps=len(rewards),
    )


def apply_actor_critic_update(
    agent: CAESARNet,
    optimizer: torch.optim.Optimizer,
    rollout: EpisodeRollout,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[float, float]:
    returns = discounted_returns(rollout.rewards, config.discount, device)
    values = torch.stack(rollout.values)
    log_probs = torch.stack(rollout.log_probs)

    advantages = returns - values.detach()
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    health_factor = 1.0 / (agent.scar.item() + 0.1)
    policy_loss = -(log_probs * advantages * health_factor).mean()
    value_loss = F.mse_loss(values, returns)

    optimizer.zero_grad(set_to_none=True)
    (policy_loss + config.value_loss_coef * value_loss).backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), config.gradient_clip_norm)
    optimizer.step()

    return float(policy_loss.item()), float(value_loss.item())


def save_training_history(history: list[TrainingSnapshot], output_path: str) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(snapshot) for snapshot in history]
    destination.write_text(json.dumps(payload, indent=2))


def train_caesar(
    config: TrainingConfig | None = None,
    env_config: EnvConfig | None = None,
    model_config: ModelConfig | None = None,
) -> list[TrainingSnapshot]:
    config = config or TrainingConfig()
    device = resolve_device(config.device)
    set_seed(config.seed)

    env = EmbodiedEnv(level=0, config=env_config)
    agent = CAESARNet(model_config=model_config).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate)
    success_window: deque[int] = deque(maxlen=config.success_window_size)

    level = 0
    history: list[TrainingSnapshot] = []

    print("--- CAESAR V12.10: NUCLEATION JUMPSTART ---")

    for episode in range(config.episodes):
        noise_scale = compute_noise_scale(episode, config)
        rollout = run_training_episode(env, agent, device, noise_scale)
        apply_actor_critic_update(agent, optimizer, rollout, config, device)
        success_window.append(1 if rollout.success else 0)

        if episode % config.log_every == 0:
            success_rate = float(np.mean(success_window)) if success_window else 0.0
            snapshot = TrainingSnapshot(
                episode=episode,
                level=level,
                success_rate=success_rate,
                success_radius=env.success_radius,
                mirror=float(agent.M_state.item()),
                scar=float(agent.scar.item()),
                total_reward=rollout.total_reward,
                steps=rollout.steps,
            )
            history.append(snapshot)
            print(
                "Ep {episode:5d} | M {mirror:.4f} | Scar {scar:.4f} | SR {sr:.2f} | "
                "Rad {radius:.2f} | Lvl {level:2d} | Ret {reward:8.2f}".format(
                    episode=episode,
                    mirror=snapshot.mirror,
                    scar=snapshot.scar,
                    sr=snapshot.success_rate,
                    radius=snapshot.success_radius,
                    level=snapshot.level,
                    reward=snapshot.total_reward,
                )
            )

            threshold = promotion_threshold(level, config)
            if success_rate >= threshold and agent.M_state.item() > 0.18:
                level += 1
                env.level = level
                success_window.clear()
                print(f"  >>> NUCLEATION SUCCESS: Level {level} <<<")

    if config.output_path:
        save_training_history(history, config.output_path)

    return history


def build_parser() -> argparse.ArgumentParser:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description="Train the CAESAR v12.10 agent.")
    parser.add_argument("--episodes", type=int, default=defaults.episodes)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path for JSON training summaries.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = TrainingConfig(
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
        output_path=args.output,
    )
    train_caesar(config=config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
