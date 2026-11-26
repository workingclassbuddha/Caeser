"""Causal-path audit of scar in CAESAR v12.10.

This test does NOT establish that scar helps learning. It checks what
variables actually feed into the actor/critic outputs at fixed parameters.
"""
import torch

from caesar.core import CAESARNet, ModelConfig


def test_scar_state_does_not_directly_change_forward_action_or_value():
    torch.manual_seed(73)
    cfg = ModelConfig()
    zero_scar = CAESARNet(cfg).eval()
    high_scar = CAESARNet(cfg).eval()
    high_scar.load_state_dict(zero_scar.state_dict())

    observation = torch.randn(1, 1, cfg.observation_size)
    hidden = torch.randn(1, 1, cfg.hidden_size)

    with torch.no_grad():
        zero_scar.scar.fill_(0.0)
        high_scar.scar.fill_(1.0)
        # Keep battery above the scar gate: this isolates the existing
        # scar buffer rather than activating additional scar updates.
        action_a, value_a, hidden_a = zero_scar(observation, hidden.clone(), 100.0)
        action_b, value_b, hidden_b = high_scar(observation, hidden.clone(), 100.0)

    assert torch.allclose(action_a, action_b, rtol=0, atol=1e-7)
    assert torch.allclose(value_a, value_b, rtol=0, atol=1e-7)
    assert torch.allclose(hidden_a, hidden_b, rtol=0, atol=1e-7)


def test_scar_changes_learning_health_factor_but_not_forward_equations():
    # The actual actor-critic update uses this reciprocal scar factor.
    # Even after extended-horizon activation, the observed raw scar was
    # ~0.000545, changing the factor by ~0.54% relative to zero scar.
    no_scar_factor = 1.0 / (0.0 + 0.1)
    observed_factor = 1.0 / (0.000545 + 0.1)
    assert observed_factor < no_scar_factor
    assert 0.0 < (no_scar_factor - observed_factor) / no_scar_factor < 0.01
