from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass(slots=True)
class EnvConfig:
    """Configuration for the embodied 2D arena."""

    base_size: float = 5.0
    size_growth_per_level: float = 1.0 / 40.0
    base_max_steps: int = 350
    max_steps_growth_per_level: int = 15
    initial_success_radius: float = 1.0
    success_radius_decay_per_level: float = 0.05
    success_radius_floor: float = 0.40
    position_margin: float = 1.0
    wall_margin: float = 0.1
    min_spawn_distance: float = 1.0
    velocity_damping: float = 0.85
    thrust_gain: float = 0.25
    position_step_scale: float = 0.12
    battery_capacity: float = 100.0
    passive_battery_drain: float = 0.02
    thrust_battery_cost: float = 0.15
    ray_count: int = 8


class EmbodiedEnv:
    """A small continuous-control arena with spatial and metabolic constraints."""

    def __init__(
        self,
        level: int = 0,
        config: EnvConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.level = level
        self.config = config or EnvConfig()
        self.rng = rng or np.random.default_rng()
        self.reset()

    def reset(self) -> np.ndarray:
        self.size = self.config.base_size + (self.level * self.config.size_growth_per_level)
        self.max_steps = self.config.base_max_steps + (
            self.level * self.config.max_steps_growth_per_level
        )
        self.success_radius = max(
            self.config.success_radius_floor,
            self.config.initial_success_radius
            - (self.level * self.config.success_radius_decay_per_level),
        )

        self.agent_pos = self._sample_position()
        self.goal_pos = self._sample_goal_position()
        self.velocity = np.zeros(2, dtype=np.float32)
        self.battery = self.config.battery_capacity
        self.steps = 0
        return self._get_obs()

    def _sample_position(self) -> np.ndarray:
        return self.rng.uniform(
            self.config.position_margin,
            self.size - self.config.position_margin,
            size=2,
        ).astype(np.float32)

    def _sample_goal_position(self) -> np.ndarray:
        minimum_distance = max(self.config.min_spawn_distance, self.success_radius * 1.5)
        for _ in range(100):
            candidate = self._sample_position()
            if np.linalg.norm(candidate - self.agent_pos) >= minimum_distance:
                return candidate
        return self._sample_position()

    def _get_obs(self) -> np.ndarray:
        delta = self.goal_pos - self.agent_pos
        distance = np.linalg.norm(delta)
        direction = delta / distance if distance > 1e-6 else np.zeros(2, dtype=np.float32)
        obs = [
            float(self.velocity[0]),
            float(self.velocity[1]),
            float(direction[0]),
            float(direction[1]),
            float(distance / self.size),
            float(self.battery / self.config.battery_capacity),
        ]

        angles = np.linspace(0, 2 * np.pi, self.config.ray_count, endpoint=False)
        for angle in angles:
            ray = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            distance_to_wall = self.size * 2.0
            if ray[0] > 0:
                distance_to_wall = min(
                    distance_to_wall,
                    (self.size - self.agent_pos[0]) / ray[0],
                )
            elif ray[0] < 0:
                distance_to_wall = min(distance_to_wall, -self.agent_pos[0] / ray[0])

            if ray[1] > 0:
                distance_to_wall = min(
                    distance_to_wall,
                    (self.size - self.agent_pos[1]) / ray[1],
                )
            elif ray[1] < 0:
                distance_to_wall = min(distance_to_wall, -self.agent_pos[1] / ray[1])

            obs.append(float(distance_to_wall / (self.size * 2.0)))

        return np.array(obs, dtype=np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, float | bool]]:
        self.steps += 1
        thrust = np.clip(action, -1.0, 1.0)
        self.velocity = self.velocity * self.config.velocity_damping + thrust * self.config.thrust_gain
        self.agent_pos += self.velocity * self.config.position_step_scale
        self.agent_pos = np.clip(
            self.agent_pos,
            self.config.wall_margin,
            self.size - self.config.wall_margin,
        )

        drain = self.config.passive_battery_drain + (
            np.linalg.norm(thrust) * self.config.thrust_battery_cost
        )
        self.battery = max(0.0, self.battery - float(drain))
        distance = np.linalg.norm(self.agent_pos - self.goal_pos)

        time_weight = 1.0 + (self.steps / 150.0)
        reward = -0.3 * float(distance) * time_weight

        success = False
        if distance < self.success_radius:
            reward += 500.0 + (self.battery * 35.0)
            success = True
            done = True
        elif self.battery <= 0 or self.steps >= self.max_steps:
            reward -= 100.0
            done = True
        else:
            done = False

        return self._get_obs(), reward, done, {
            "success": success,
            "distance": float(distance),
            "battery": float(self.battery),
        }


@dataclass(slots=True)
class ModelConfig:
    """Configuration for the recurrent actor-critic network."""

    observation_size: int = 14
    hidden_size: int = 128
    action_size: int = 2
    initial_log_std: float = -1.0
    mirror_momentum: float = 0.99
    scar_threshold: float = 0.12
    scar_battery_threshold: float = 5.0
    scar_increment: float = 0.000005


class CAESARNet(nn.Module):
    """A recurrent actor-critic with persistent mirror and scar state."""

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        scar_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.config = model_config or ModelConfig()
        self.scar_decay = scar_decay

        self.embed = nn.Sequential(
            nn.Linear(self.config.observation_size, self.config.hidden_size),
            nn.LayerNorm(self.config.hidden_size),
            nn.ReLU(),
        )
        self.gru = nn.GRU(self.config.hidden_size, self.config.hidden_size, batch_first=True)
        self.actor = nn.Linear(self.config.hidden_size, self.config.action_size)
        self.critic = nn.Linear(self.config.hidden_size, 1)
        self.log_std = nn.Parameter(
            torch.full((self.config.action_size,), self.config.initial_log_std)
        )

        self.register_buffer("scar", torch.zeros(1))
        self.register_buffer("M_state", torch.ones(1))

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: torch.Tensor,
        battery: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.embed(x)
        lattice_sharpness = 1.0 / (1.0 + torch.var(z, unbiased=False) + 1e-6)
        battery_norm = torch.as_tensor(
            max(battery, 0.0) / 100.0,
            dtype=z.dtype,
            device=z.device,
        )
        mirror = (lattice_sharpness * battery_norm).clamp_min(1e-6)

        with torch.no_grad():
            self.M_state.mul_(self.config.mirror_momentum).add_(
                (1.0 - self.config.mirror_momentum) * mirror.detach()
            )
            if self.M_state.item() < self.config.scar_threshold and battery < self.config.scar_battery_threshold:
                self.scar.add_(self.config.scar_increment)
            if self.scar_decay > 0.0:
                self.scar.mul_(max(0.0, 1.0 - self.scar_decay))

        z_out, hidden_state = self.gru(z, hidden_state)
        action_mean = torch.tanh(self.actor(z_out))
        state_value = self.critic(z_out)
        return action_mean, state_value, hidden_state
