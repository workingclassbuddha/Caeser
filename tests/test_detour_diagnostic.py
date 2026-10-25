"""Causal hygiene and deterministic inputs for v2 detour audit."""
import numpy as np
import pytest

from caesar.detour_diagnostic import (
    WallScanner, CONTROLS, run_audit, trial_rollout,
)
from caesar.challenge_benchmark import evaluate


def test_wall_scanner_accepts_observations_only():
    scanner = WallScanner()
    obs = np.zeros(14, dtype=np.float32)
    obs[2] = 1.0
    obs[6] = 0.04  # east ray hits a nearby wall
    obs[8] = 0.35  # room toward the north
    action = scanner.action(obs)
    assert scanner.phase == "search_up"
    np.testing.assert_array_equal(action, [0.0, 1.0])
    assert np.max(abs(action)) <= 1.0
    # Direction and ray observations, not unseen obstacle coordinates,
    # drive the phase transition.
    obs[6] = 0.2
    action = scanner.action(obs)
    assert scanner.phase == "cross"
    np.testing.assert_array_equal(action, [1.0, 0.0])
    with pytest.raises(ValueError):
        scanner.action(np.zeros(10, dtype=np.float32))


def test_wall_scanner_reverses_at_visible_northern_boundary():
    scanner = WallScanner(phase="search_up")
    obs = np.zeros(14, dtype=np.float32)
    obs[6] = 0.02  # still blocked to east
    obs[8] = 0.01  # top boundary now nearby
    action = scanner.action(obs)
    assert scanner.phase == "search_down"
    np.testing.assert_array_equal(action, [0.0, -1.0])


@pytest.mark.parametrize("controller", ["direct", "oracle_route"])
def test_audit_trials_match_existing_evaluation_success_and_episode_length(controller):
    seed = 11
    episodes = [trial_rollout(seed, "none", trial, controller) for trial in range(2)]
    expected = evaluate(seed, "none", 2, controller=controller)
    assert [x["success"] for x in episodes] == [
        row["success"] for row in expected["trials_log"]
    ]
    assert [x["steps"] for x in episodes] == [
        row["steps"] for row in expected["trials_log"]
    ]


def test_oracle_route_can_sacrifice_one_step_distance_reduction():
    # A proper one-step counterfactual, not a claim about learned reasoning.
    sample = [trial_rollout(11, "none", i, "oracle_route") for i in range(4)]
    assert all(x["success"] for x in sample)
    assert any(x["one_step_direct_better_count"] > 0 for x in sample)
    assert any(x["sum_one_step_distance_sacrifice"] > 0 for x in sample)
    direct = trial_rollout(11, "none", 0, "direct")
    assert direct["one_step_direct_better_count"] == 0


def test_audit_reports_all_controllers_without_training():
    result = run_audit([11], 2, ["none", "space+time+energy"])
    assert set(result["summary"]) == set(CONTROLS)
    assert len(result["trials"]) == 2 * 2 * len(CONTROLS)
    assert result["summary"]["direct"]["total_trials"] == 4
    assert result["summary"]["oracle_route"]["success_rate"] == 1.0
    assert result["summary"]["direct"]["success_rate"] == 0.0
    assert result == run_audit([11], 2, ["none", "space+time+energy"])


def test_rejects_unsupported_protocol_inputs():
    with pytest.raises(ValueError):
        run_audit([11, 11], 1, ["none"])
    with pytest.raises(ValueError):
        run_audit([11], 1, ["none", "none"])
    with pytest.raises(ValueError):
        trial_rollout(11, "none", -1, "direct")
