import numpy as np
import torch

from caesar.core import ModelConfig
from caesar.observer import ReflexiveCAESARNet, measure_observer_order


def test_observer_metrics_are_bounded():
    rng = np.random.default_rng(42)
    hidden = np.cumsum(rng.normal(size=(64, 16)) * 0.05, axis=0)
    predicted = hidden + rng.normal(size=hidden.shape) * 0.01
    metrics = measure_observer_order(hidden, predicted)
    for value in metrics.as_dict().values():
        assert 0.0 <= value <= 1.0


def test_reflexive_forward_shapes():
    agent = ReflexiveCAESARNet(ModelConfig(observation_size=14, hidden_size=32), recursive=True)
    x = torch.randn(1, 1, 14)
    h = torch.zeros(1, 1, 32)
    action, value, hidden, self_state, predicted = agent(x, h, 100.0)
    assert action.shape == (1, 1, 2)
    assert value.shape == (1, 1, 1)
    assert hidden.shape == (1, 1, 32)
    assert self_state.shape == (1, 1, 32)
    assert predicted.shape == (1, 1, 32)
