from caesar.scar_probe import probe


def test_scar_probe_is_reproducible():
    first = probe(seed=91, episodes=1)
    second = probe(seed=91, episodes=1)
    assert first == second
    assert first["steps"] > 0
    assert first["scar_increment_events"] <= first["scar_eligible_steps"]
    assert first["final_scar"] >= 0


def test_scar_probe_rejects_invalid_conditions():
    import pytest
    with pytest.raises(ValueError):
        probe(episodes=0)
    with pytest.raises(ValueError):
        probe(level=-1)


def test_scar_positive_control_is_reachable():
    # This is an artificial, deliberately stressed condition, NOT a normal
    # environment episode or evidence of improved learning.
    import torch
    from caesar.core import CAESARNet

    model = CAESARNet().eval()
    with torch.no_grad():
        model.M_state.fill_(0.10)
        x = torch.zeros(1, 1, model.config.observation_size)
        h = torch.zeros(1, 1, model.hidden_size)
        model(x, h, battery=2.0)
    assert abs(model.scar.item() - model.config.scar_increment) < 1e-9


def test_scar_negative_control_remains_zero_with_full_battery():
    import torch
    from caesar.core import CAESARNet

    model = CAESARNet().eval()
    with torch.no_grad():
        model.M_state.fill_(0.10)
        x = torch.zeros(1, 1, model.config.observation_size)
        h = torch.zeros(1, 1, model.hidden_size)
        model(x, h, battery=100.0)
    assert model.scar.item() == 0.0
