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
