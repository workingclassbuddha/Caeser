"""Controls for same-learning-rule physical constraint curriculum."""
import numpy as np
import pytest

from caesar.challenge_curriculum import (
    HARD_HALF_WIDTH, WIDE_HALF_WIDTH, width_for_episode, run_curriculum,
)
from caesar.challenge_env import BarrierEnv
from caesar.challenge_benchmark import evaluate


def test_graduated_schedule_only_tightens_passage_width():
    widths = [width_for_episode("graduated", i, 20) for i in range(20)]
    assert widths[0] == WIDE_HALF_WIDTH
    assert widths[-1] == HARD_HALF_WIDTH
    assert all(a >= b for a, b in zip(widths, widths[1:]))
    assert all(width_for_episode("fixed_hard", i, 20) == HARD_HALF_WIDTH
               for i in range(20))
    with pytest.raises(ValueError):
        width_for_episode("invented", 0, 20)


def test_wide_opening_passes_straight_trajectory_hard_blocks_it():
    env = BarrierEnv(rng=np.random.default_rng(13), fixed_gap_side="top")
    mid_y = 2.5
    assert not env._gap_open(mid_y)
    env.gap_half_width = WIDE_HALF_WIDTH
    assert env._gap_open(mid_y)
    env.gap_half_width = HARD_HALF_WIDTH
    assert not env._gap_open(mid_y)


def test_test_policy_is_always_full_hard_regardless_of_training_schedule():
    reference = evaluate(seed=11, arm="none", trials=2, controller="oracle_route")
    direct = evaluate(seed=11, arm="none", trials=2, controller="direct")
    assert reference["successes"] == 2
    assert direct["successes"] == 0


def test_both_schedules_run_with_matched_seeds_and_preserve_logs():
    result = run_curriculum([11], episodes=2, trials=1)
    assert set(result["summary"]) == {"fixed_hard", "graduated"}
    assert len(result["runs"]) == 2
    assert result["runs"][0]["training_trace"][0]["gap_half_width"] == HARD_HALF_WIDTH
    assert result["runs"][1]["training_trace"][0]["gap_half_width"] == WIDE_HALF_WIDTH
    assert len(result["runs"][1]["training_trace"]) == 2
    assert result["runs"][0]["pretrain"] == result["runs"][1]["pretrain"]
    assert result["runs"][0]["training_trace"][0]["gap_half_width"] != (
        result["runs"][1]["training_trace"][0]["gap_half_width"]
    )


def test_invalid_schedule_arguments():
    with pytest.raises(ValueError):
        run_curriculum([11, 11], 5, 2)
