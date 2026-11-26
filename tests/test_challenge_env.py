"""Regress actual obstacle physics, ray sensing and challenge evaluation."""
import numpy as np
import pytest
import torch
from dataclasses import asdict

from caesar.challenge_env import BarrierEnv, CHALLENGE_ARMS, challenge_config
from caesar.challenge_benchmark import evaluate, run_challenge
from caesar.core import CAESARNet


def test_factorial_tightening_changes_only_declared_constraints():
    base = asdict(challenge_config("none"))
    a = asdict(challenge_config("space"))
    b = asdict(challenge_config("time"))
    c = asdict(challenge_config("energy"))
    def differences(target):
        return {k for k in base if base[k] != target[k]}
    assert differences(a) == {"initial_success_radius", "success_radius_floor"}
    assert differences(b) == {"base_max_steps"}
    assert differences(c) == {"thrust_battery_cost"}
    combined = asdict(challenge_config("space+time+energy"))
    assert differences(combined) == {
        "initial_success_radius", "success_radius_floor",
        "base_max_steps", "thrust_battery_cost"
    }
    assert len(CHALLENGE_ARMS) == 8


@pytest.mark.parametrize("gap", ["top", "bottom"])
def test_wall_blocks_crossing_outside_gap_without_teleporting(gap):
    env = BarrierEnv(rng=np.random.default_rng(17), fixed_gap_side=gap)
    env.agent_pos = np.array([env.wall_x - 0.02, 2.50], dtype=np.float32)
    env.velocity[:] = 0
    obs_before = env._get_obs()
    # Ray aimed directly right senses the barrier rather than outer wall.
    assert obs_before[6] < 0.01
    _, _, _, info = env.step(np.array([1.0, 0.0]))
    assert info["collided"] is True
    assert env.agent_pos[0] < env.wall_x
    assert env.collision_count == 1


@pytest.mark.parametrize("gap", ["top", "bottom"])
def test_passage_allows_crossing_only_at_gap(gap):
    env = BarrierEnv(rng=np.random.default_rng(29), fixed_gap_side=gap)
    env.agent_pos = np.array([env.wall_x - 0.01, env.gap_center], dtype=np.float32)
    env.velocity[:] = 0
    before = env._get_obs()
    # Directly right, wall has a gap. The outer wall remains visible.
    assert before[6] > 0.1
    _, _, _, info = env.step(np.array([1.0, 0.0]))
    assert not info["collided"]
    assert env.agent_pos[0] > env.wall_x


def test_eval_same_seed_equal_geometry_and_frozen_model_state():
    torch.manual_seed(11)
    agent = CAESARNet()
    before = {key: value.clone() for key, value in agent.state_dict().items()}
    a = evaluate(11, "none", 2, controller=None, agent=agent)
    b = evaluate(11, "none", 2, controller=None, agent=agent)
    assert a == b
    for key, value in agent.state_dict().items():
        assert torch.equal(value, before[key])


def test_direct_policy_is_not_sufficient_on_wall_gap_and_stationary_fails():
    # Frozen small sample, not a claim about every possible layout.
    for gap_result in (evaluate(11, "none", 2, controller="direct"),
                       evaluate(11, "none", 2, controller="velocity_compensated"),
                       evaluate(11, "none", 2, controller="stationary")):
        assert gap_result["successes"] == 0


def test_privileged_reference_route_is_reachable_in_standard_layout():
    ref = evaluate(11, "none", 2, controller="oracle_route")
    assert ref["successes"] >= 1, "Feasibility control must find a path at least once"


def test_full_agent_training_pipeline_records_pre_post_and_controls():
    out = run_challenge([11], episodes=1, trials=1, train_arms=["none"])
    assert out["protocol"] == "caesar_v2_barrier_constraints_v1"
    assert len(out["training"]) == 1
    assert len(out["training"][0]["train_log"]) == 1
    assert set(out["training"][0]["posttrain"]) == set(CHALLENGE_ARMS)
    assert "direct" in out["controller_summary"]
