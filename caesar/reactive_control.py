"""Predeclared, nonlearning v2 wall-scanning policy using only CAESAR's observation."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

@dataclass
class WallScanner:
    """Fixed policy that follows the visible wall, exploring upward first.

    No global map, gap flag, coordinates, reward or simulator internals:
    only the 14-element CAESAR observation and a four-state finite automaton.
    Thresholds and 15-step wall crossing are frozen *a priori*.
    """
    phase: str = "approach"
    crossing_steps: int = 0
    phase_counts: dict[str, int] = field(default_factory=dict)

    def action(self, obs: np.ndarray) -> np.ndarray:
        if obs.shape != (14,):
            raise ValueError("expected CAESAR's 14-element observation")
        east_ray = float(obs[6])
        north_ray = float(obs[8])
        velocity = obs[0:2]
        goal_direction = obs[2:4]
        if self.phase == "approach" and east_ray < 0.065:
            self.phase = "search_up"
        if self.phase == "search_up":
            if east_ray > 0.095:
                self.phase = "cross"
                self.crossing_steps = 0
            elif north_ray < 0.06:
                self.phase = "search_down"
        if self.phase == "search_down" and east_ray > 0.095:
            self.phase = "cross"
            self.crossing_steps = 0
        if self.phase == "cross":
            self.crossing_steps += 1
            if self.crossing_steps >= 18:
                self.phase = "finish"
        self.phase_counts[self.phase] = self.phase_counts.get(self.phase, 0) + 1
        if self.phase == "search_up":
            desired = np.array([0.0, 1.0], dtype=np.float32)
        elif self.phase == "search_down":
            desired = np.array([0.0, -1.0], dtype=np.float32)
        elif self.phase == "cross":
            desired = np.array([1.0, 0.0], dtype=np.float32)
        else:
            desired = goal_direction.astype(np.float32)
        return np.clip(desired - velocity, -1, 1).astype(np.float32)

