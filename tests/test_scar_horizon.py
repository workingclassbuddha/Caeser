from caesar.trained_scar_probe import trained_probe


def test_explicit_horizon_changes_episode_budget_only():
    result = trained_probe(seed=17, episodes=1, max_steps=1)
    assert result["base_max_steps"] == 1
    assert result["horizon_override"] is True
    assert result["total_steps"] == 1


def test_invalid_horizon_rejected():
    import pytest
    with pytest.raises(ValueError):
        trained_probe(seed=17, episodes=1, max_steps=0)
