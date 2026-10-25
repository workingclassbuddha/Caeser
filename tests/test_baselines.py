"""Deterministic, observation-only CAESAR control baselines."""
import numpy as np
import pytest

from caesar.baselines import (
    CONTROLLERS, controller_action, evaluate_controller, run_baselines,
)
from caesar.constraint_transfer import ARMS, arm_config
from caesar.core import EmbodiedEnv


def test_fixed_controller_uses_observed_target_direction_and_velocity():
    obs = np.zeros(14, dtype=np.float32)
    obs[2:4] = [1.0, 0.0]
    obs[0:2] = [0.3, -0.2]
    np.testing.assert_allclose(controller_action(obs, "direct"), [1.0, 0.0])
    np.testing.assert_allclose(controller_action(obs, "velocity_compensated"), [0.7, 0.2])
    np.testing.assert_allclose(controller_action(obs, "stationary"), [0.0, 0.0])
    with pytest.raises(ValueError):
        controller_action(obs, "unknown")


def test_baselines_never_access_oracle_goal_and_are_action_bounded():
    obs = np.array([0.8, -0.9, 0.9, -0.9] + [0.0] * 10, dtype=np.float32)
    for controller in CONTROLLERS:
        a = controller_action(obs, controller)
        assert a.shape == (2,)
        assert (np.abs(a) <= 1).all()


def test_matched_initial_geometry_same_across_all_constraint_arms():
    # The evaluator's first explicit reset after construction must remain
    # matched across arms so changes are due to the constraints, not the goals.
    initial = []
    for arm in ARMS:
        env = EmbodiedEnv(
            level=0, config=arm_config(arm),
            rng=np.random.default_rng(100_000 + 11 * 10_000),
        )
        env.reset()
        initial.append((env.agent_pos.copy(), env.goal_pos.copy()))
    for positions in initial[1:]:
        np.testing.assert_array_equal(positions[0], initial[0][0])
        np.testing.assert_array_equal(positions[1], initial[0][1])


def test_baseline_reproducible_and_reports_cross_condition_controls():
    first = run_baselines([11], 1)
    second = run_baselines([11], 1)
    assert first == second
    assert set(first["summary"]) == set(CONTROLLERS)
    assert set(first["evaluations"]["direct"]["11"]) == set(ARMS)
    for controller in CONTROLLERS:
        for arm in ARMS:
            rate = first["summary"][controller]["test_success_by_arm"][arm]
            assert 0 <= rate <= 1
            assert 0 <= first["summary"][controller][
                "cross_condition_success_excluding_training_arm"
            ][arm] <= 1


def test_invalid_baseline_arguments_fail_closed():
    with pytest.raises(ValueError):
        run_baselines([11, 11], 1)
    with pytest.raises(ValueError):
        evaluate_controller("direct", "space", 11, 0)
    with pytest.raises(ValueError):
        evaluate_controller("bad", "space", 11, 1)
