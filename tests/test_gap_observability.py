"""Verify paired-world observability audit does not confound geometry."""
import numpy as np
import pytest

from caesar.gap_observability import pair, run_audit


def test_only_gap_side_differs_in_paired_world_observations():
    row = pair(seed=11, trial=0)
    assert len(row["observations"]) == 2
    assert row["observations"][0]["stage"] == "spawn"
    assert row["observations"][1]["stage"] == "near_wall"
    for stage in row["observations"]:
        assert stage["nonray_max_delta"] <= 1e-6
        assert len(stage["top_sensor_rays"]) == 8
        assert len(stage["bottom_sensor_rays"]) == 8
        assert stage["ray_max_delta"] >= 0
    assert row == pair(seed=11, trial=0)


def test_near_wall_approach_moves_but_never_crosses_blocked_wall():
    row = pair(seed=11, trial=1)
    assert row["observations"][1]["approach_steps"] > 0
    assert row["initial_position"][0] < row["near_wall_position"][0]
    assert row["near_wall_position"][0] < 2.5


def test_multiple_seeds_are_summarized_without_ground_truth_leakage():
    out = run_audit([11, 23], trials=2)
    assert out["paired_worlds"] == 4
    assert len(out["trials"]) == 4
    assert set(out["summary"]) == {"spawn", "near_wall"}
    for stage in ("spawn", "near_wall"):
        assert out["summary"][stage]["nonray_values_match"]
        assert 0 <= out["summary"][stage]["fraction_sensor_distinguishable_exact"] <= 1
    with pytest.raises(ValueError):
        run_audit([11,11], trials=2)
