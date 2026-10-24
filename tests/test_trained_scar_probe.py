from caesar.trained_scar_probe import trained_probe


def test_trained_probe_runs_a_real_update_and_reports_no_fabricated_events():
    result = trained_probe(seed=31, episodes=1, level=0)
    assert result["total_steps"] > 0
    assert result["episodes"] == 1
    assert len(result["records"]) == 1
    assert result["total_scar_events_estimate"] >= 0
    assert result["episodes_with_scar"] <= 1
    assert result["final_scar"] >= 0
    assert abs(
        result["final_scar"] -
        result["total_scar_events_estimate"] * 0.000005
    ) < 0.000001


def test_trained_probe_rejects_invalid_episodes():
    import pytest
    with pytest.raises(ValueError):
        trained_probe(episodes=0)
