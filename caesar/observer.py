from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .core import CAESARNet, ModelConfig


@dataclass(slots=True)
class ObserverMetrics:
    """Operational proxies; these are not a claim of subjective experience."""

    integration: float
    temporal_persistence: float
    causal_closure: float
    self_prediction: float
    irreducibility: float
    reflexivity: float
    observer_order: float

    def as_dict(self) -> dict[str, float]:
        return {
            "integration": self.integration,
            "temporal_persistence": self.temporal_persistence,
            "causal_closure": self.causal_closure,
            "self_prediction": self.self_prediction,
            "irreducibility": self.irreducibility,
            "reflexivity": self.reflexivity,
            "observer_order": self.observer_order,
        }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    return float(np.clip(abs(value), 0.0, 1.0))


def _normalized_entropy(values: np.ndarray) -> float:
    singular = np.linalg.svd(values - values.mean(axis=0, keepdims=True), compute_uv=False)
    if len(singular) == 0 or singular.sum() < 1e-12:
        return 0.0
    p = singular / singular.sum()
    entropy = -float(np.sum(p * np.log(p + 1e-12)))
    return float(np.clip(entropy / np.log(len(p)), 0.0, 1.0)) if len(p) > 1 else 0.0


def measure_observer_order(
    hidden_states: np.ndarray,
    self_predictions: np.ndarray | None = None,
    next_hidden_states: np.ndarray | None = None,
    ablated_predictions: np.ndarray | None = None,
) -> ObserverMetrics:
    """Measure a falsifiable dynamical signature of observer-like organization.

    The score intentionally uses measurable proxies rather than labeling a system
    conscious. It is designed for comparing CAESAR variants under matched seeds.
    """
    h = np.asarray(hidden_states, dtype=float)
    if h.ndim != 2 or len(h) < 4:
        raise ValueError("hidden_states must have shape [time, features] with at least 4 rows")

    # Integration proxy: broad participation of the hidden state, not raw magnitude.
    integration = _normalized_entropy(h)

    # Persistence: adjacent global-state similarity.
    temporal_persistence = float(
        np.mean([_safe_corr(h[i], h[i + 1]) for i in range(len(h) - 1)])
    )

    # Causal-closure proxy: predict the next global state from the current state.
    x = h[:-1]
    y = h[1:]
    closure = np.mean([_safe_corr(x[:, i], y[:, i]) for i in range(h.shape[1])])
    causal_closure = float(np.clip(closure, 0.0, 1.0))

    if self_predictions is None:
        self_prediction = 0.0
        reflexivity = 0.0
    else:
        pred = np.asarray(self_predictions, dtype=float)
        target = h[: len(pred)]
        self_prediction = float(
            np.mean([_safe_corr(pred[:, i], target[:, i]) for i in range(min(pred.shape[1], target.shape[1]))])
        )
        # Reflexivity is explicitly tied to whether the self-model adds predictive
        # information beyond a persistence baseline.
        baseline = temporal_persistence
        reflexivity = float(np.clip(self_prediction - baseline + 0.5, 0.0, 1.0))

    if ablated_predictions is None or self_predictions is None:
        irreducibility = 0.0
    else:
        full = np.asarray(self_predictions, dtype=float)
        ablated = np.asarray(ablated_predictions, dtype=float)
        target = h[: len(full)]
        full_err = float(np.mean((full - target) ** 2)) + 1e-9
        ablated_err = float(np.mean((ablated - target) ** 2))
        irreducibility = float(np.clip((ablated_err - full_err) / (ablated_err + 1e-9), 0.0, 1.0))

    observer_order = float(
        np.clip(
            (
                integration
                * temporal_persistence
                * causal_closure
                * max(self_prediction, 1e-6)
                * max(irreducibility, 1e-6)
            )
            ** (1.0 / 5.0),
            0.0,
            1.0,
        )
    )

    return ObserverMetrics(
        integration=integration,
        temporal_persistence=temporal_persistence,
        causal_closure=causal_closure,
        self_prediction=self_prediction,
        irreducibility=irreducibility,
        reflexivity=reflexivity,
        observer_order=observer_order,
    )


class ReflexiveCAESARNet(nn.Module):
    """CAESAR with a learned model of its own recent hidden state.

    The base CAESAR weights are preserved; the self-model is the only added
    trainable pathway. This makes Control vs Reflexive comparisons explicit.
    """

    def __init__(self, model_config: ModelConfig | None = None, recursive: bool = False) -> None:
        super().__init__()
        self.base = CAESARNet(model_config=model_config)
        hidden = self.base.hidden_size
        self.recursive = recursive
        self.self_model = nn.GRU(hidden, hidden, batch_first=True)
        self.self_head = nn.Linear(hidden, hidden)
        self.fusion = nn.Linear(hidden * (2 if recursive else 1), hidden)

    @property
    def hidden_size(self) -> int:
        return self.base.hidden_size

    @property
    def M_state(self) -> torch.Tensor:
        return self.base.M_state

    @property
    def scar(self) -> torch.Tensor:
        return self.base.scar

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: torch.Tensor,
        battery: float,
        self_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean, value, hidden = self.base(x, hidden_state, battery)
        detached_history = hidden.detach()
        if self_state is None:
            self_state = torch.zeros(1, 1, self.hidden_size, device=x.device, dtype=x.dtype)
        predicted_self, self_state = self.self_model(detached_history, self_state)
        predicted_self = self.self_head(predicted_self)
        if self.recursive:
            fused = torch.tanh(self.fusion(torch.cat([hidden, predicted_self], dim=-1)))
            action_mean = torch.tanh(self.base.actor(fused))
            value = self.base.critic(fused)
        return action_mean, value, hidden, self_state, predicted_self
