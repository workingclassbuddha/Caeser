"""Validate causal memory intervention without mistaking numerical contrast for skill."""
import numpy as np
import pytest
import torch

from caesar.core import CAESARNet
from caesar.memory_causal_probe import (
    StoredRayBitScanner, paired_histories, replay_policy, probe_model,
    evaluate_scanner, run_audit,
)
from caesar.challenge_benchmark import evaluate


def test_paired_worlds_have_identical_final_input_in_a_subset():
    sample = [paired_histories(11, trial) for trial in range(12)]
    assert any(item["same_current_obs"] for item in sample)
    for item in sample:
        a, b = item["histories"]["top"], item["histories"]["bottom"]
        assert len(a) == len(b) and len(a) >= 2
        np.testing.assert_array_equal(a[0][:6], b[0][:6])
        if item["same_current_obs"]:
            assert np.max(np.abs(a[-1] - b[-1])) < 1e-6


def test_exact_same_last_observation_after_forced_memory_reset_gives_identical_action():
    torch.manual_seed(11)
    agent = CAESARNet()
    pair = next(
        paired_histories(11, index) for index in range(12)
        if paired_histories(11, index)["same_current_obs"]
    )
    current = pair["histories"]["top"][-1]
    history_a = pair["histories"]["top"]
    history_b = pair["histories"]["bottom"]
    assert np.max(np.abs(current - history_b[-1])) < 1e-6
    a, _ = replay_policy(agent, history_a, reset_hidden=True)
    b, _ = replay_policy(agent, history_b, reset_hidden=True)
    np.testing.assert_allclose(a, b, atol=1e-6)
    state_before = {name: val.clone() for name,val in agent.state_dict().items()}
    replay_policy(agent, history_a, reset_hidden=False)
    for name, val in agent.state_dict().items():
        assert torch.equal(val, state_before[name]), name


def test_frozen_bit_policy_uses_first_observation_not_gap_ground_truth():
    scanner = StoredRayBitScanner()
    obs = np.zeros(14, dtype=np.float32)
    obs[7], obs[13] = 0.20, 0.08
    obs[6] = 0.3
    obs[8] = 0.3  # room remains toward the north
    scanner.action(obs)
    assert scanner.preferred_side == "top"
    obs[7], obs[13], obs[6] = 0.0, 1.0, 0.02
    scanner.action(obs)
    assert scanner.preferred_side == "top"
    assert scanner.phase == "search_up"


def test_one_bit_and_default_share_test_cases_and_report_full_log():
    sample = evaluate_scanner(11, 3, "none", "stored_ray_bit")
    default = evaluate_scanner(11, 3, "none", "default_up")
    assert sample["trials"] == default["trials"]
    assert [x["actual_gap"] for x in sample["trial_log"]] == [
        x["actual_gap"] for x in default["trial_log"]
    ]
    expected = evaluate(11, "none", 3, controller="wall_scanner")
    assert [x["success"] for x in default["trial_log"]] == [
        x["success"] for x in expected["trials_log"]
    ]


def test_end_to_end_probe_reports_per_pair_causal_controls():
    result = run_audit([11], train_episodes=0, probe_trials=2, control_trials=1)
    assert result["protocol"] == "caesar_v2_frozen_gru_history_reset_v1"
    row = result["learners"][0]
    assert row["untrained"]["pairs_total"] == 2
    assert row["trained"]["pairs_total"] == 2
    assert set(result["control_summary"]) == {"default_up", "stored_ray_bit"}
    with pytest.raises(ValueError):
        run_audit([11, 11], 0, 2, 1)
