"""CAESAR Challenge Arena v2: a solid dividing wall with a reachable gap.

A new, separately versioned task. The existing v1 arena and learning
architecture are untouched. Goal direction remains observable; direct
goal steering is no longer sufficient because the shortest line is blocked.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .core import EmbodiedEnv, EnvConfig

CHALLENGE_ARMS = (
    "none", "space", "time", "energy", "space+time",
    "space+energy", "time+energy", "space+time+energy",
)
GAP_HALF_WIDTH = 0.55
WALL_CLEARANCE = 0.015


def challenge_config(arm: str = "none") -> EnvConfig:
    if arm not in CHALLENGE_ARMS:
        raise ValueError("unknown challenge arm")
    factors = set() if arm == "none" else set(arm.split("+"))
    base = EnvConfig()
    # All arms have exactly the same geometry; only three independent
    # physical constraint strengths change.
    return replace(
        base,
        initial_success_radius=0.30 if "space" in factors else 0.50,
        success_radius_floor=0.30 if "space" in factors else 0.50,
        base_max_steps=105 if "time" in factors else 190,
        battery_capacity=50.0,
        thrust_battery_cost=0.36 if "energy" in factors else 0.15,
        min_spawn_distance=1.5,
    )


class BarrierEnv(EmbodiedEnv):
    """Two-sided goals with a solid wall except one hidden-to-policy passage.

    Goal bearing, velocity, battery and eight ray sensors use the original
    14-dimensional CAESAR observation. There is no global route oracle in
    the agent's observation.
    """

    def __init__(
        self, level: int = 0, config: EnvConfig | None = None,
        rng: np.random.Generator | None = None, fixed_gap_side: str | None = None,
        gap_half_width: float = GAP_HALF_WIDTH,
    ) -> None:
        if fixed_gap_side not in (None, "bottom", "top"):
            raise ValueError("gap must be bottom, top, or random")
        if not GAP_HALF_WIDTH <= gap_half_width <= 2.1:
            raise ValueError("gap_half_width must lie between hard-gap width and 2.1")
        self.fixed_gap_side = fixed_gap_side
        self.gap_half_width = float(gap_half_width)
        super().__init__(level=level, config=config or challenge_config(), rng=rng)

    def reset(self) -> np.ndarray:
        size = self.config.base_size + self.level * self.config.size_growth_per_level
        self.wall_x = size * 0.5
        self.gap_side = (
            self.fixed_gap_side
            if self.fixed_gap_side is not None
            else ("bottom" if int(self.rng.integers(0, 2)) == 0 else "top")
        )
        self.gap_center = 1.15 if self.gap_side == "bottom" else size - 1.15
        self.collision_count = 0
        return super().reset()

    def _sample_position(self) -> np.ndarray:
        # Left-of-wall start; y-coordinate always opposite both openings,
        # so a straight action toward the right-hand goal is blocked.
        return self.rng.uniform([0.60, 2.25], [1.05, 2.75]).astype(np.float32)

    def _sample_goal_position(self) -> np.ndarray:
        return self.rng.uniform([self.size - 1.05, 2.25],
                                [self.size - 0.60, 2.75]).astype(np.float32)

    def _gap_open(self, crossing_y: float) -> bool:
        return abs(crossing_y - self.gap_center) <= self.gap_half_width

    def _get_obs(self) -> np.ndarray:
        obs = super()._get_obs()
        origin = self.agent_pos
        for index, theta in enumerate(np.linspace(0, 2 * np.pi, self.config.ray_count, endpoint=False)):
            dx, dy = float(np.cos(theta)), float(np.sin(theta))
            if abs(dx) < 1e-8:
                continue
            ray_dist = (self.wall_x - float(origin[0])) / dx
            if ray_dist <= 0:
                continue
            y_intersection = float(origin[1]) + ray_dist * dy
            if not self._gap_open(y_intersection):
                obs[6 + index] = min(float(obs[6 + index]), ray_dist / (self.size * 2.0))
        return obs

    def step(self, action: np.ndarray):
        # Identical drive dynamics to v1; block physical wall crossings.
        self.steps += 1
        thrust = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        before = self.agent_pos.copy()
        self.velocity = self.velocity * self.config.velocity_damping + thrust * self.config.thrust_gain
        after = np.clip(
            before + self.velocity * self.config.position_step_scale,
            self.config.wall_margin,
            self.size - self.config.wall_margin,
        )
        blocked = False
        dx = float(after[0] - before[0])
        if abs(dx) > 1e-9 and min(before[0], after[0]) <= self.wall_x <= max(before[0], after[0]):
            fraction = (self.wall_x - float(before[0])) / dx
            crossing_y = float(before[1]) + fraction * float(after[1] - before[1])
            if not self._gap_open(crossing_y):
                blocked = True
                after[0] = (
                    self.wall_x - WALL_CLEARANCE
                    if before[0] < self.wall_x else self.wall_x + WALL_CLEARANCE
                )
                self.velocity[0] = 0.0
                self.collision_count += 1
        self.agent_pos = after.astype(np.float32)
        drain = self.config.passive_battery_drain + (
            np.linalg.norm(thrust) * self.config.thrust_battery_cost
        )
        self.battery = max(0.0, self.battery - float(drain))
        distance = np.linalg.norm(self.agent_pos - self.goal_pos)
        reward = -0.3 * float(distance) * (1.0 + self.steps / 150.0)
        success = bool(distance < self.success_radius)
        if success:
            reward += 500.0 + self.battery * 35.0
            done = True
        elif self.battery <= 0.0 or self.steps >= self.max_steps:
            reward -= 100.0
            done = True
        else:
            done = False
        info = {
            "success": success,
            "distance": float(distance),
            "battery": float(self.battery),
            "collided": blocked,
            "collisions": int(self.collision_count),
        }
        return self._get_obs(), float(reward), done, info
