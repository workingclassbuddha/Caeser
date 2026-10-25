"""Paired-trial attribution check."""
from copy import deepcopy

import pytest

from caesar.baselines import run_baselines
from caesar.constraint_transfer import ARMS, run_study
from caesar.control_comparison import compare


def test_paired_comparison_matches_seed_and_trial_outcomes():
    learned = run_study([11], episodes=1, eval_trials=1, arms=["none"])
    controls = run_baselines([11], trials=1)
    report = compare(learned, controls)
    assert report["protocol"] == "caesar_paired_no_learning_comparison_v1"
    assert report["train_seeds"] == [11]
    assert len(report["comparison"]["none"]["direct"]["per_training_seed"]) == 1
    assert report["comparison"]["none"]["direct"]["per_training_seed"][0]["paired_test_count"] == 7
    assert set(controls["summary"]["direct"]["test_success_by_arm"]) == set(ARMS)

    # Make the learned outcomes identical to the direct controller by trial,
    # then the comparison must be identically zero.
    for arm in ARMS:
        row = learned["runs"][0]["end_eval"][arm]
        truth = controls["evaluations"]["direct"]["11"][arm]["trial_success"]
        row["trial_success"] = truth.copy()
    same = compare(learned, controls)
    assert same["comparison"]["none"]["direct"]["mean_seed_paired_success_delta"] == 0.0


def test_mismatched_seeds_or_trials_cannot_be_averaged():
    learned = run_study([11], episodes=1, eval_trials=1, arms=["none"])
    controls = run_baselines([11], trials=1)
    wrong = deepcopy(controls)
    wrong["seeds"] = [99]
    with pytest.raises(ValueError, match="Seed mismatches"):
        compare(learned, wrong)
    wrong = deepcopy(controls)
    wrong["heldout_trials_per_test_arm"] = 2
    with pytest.raises(ValueError, match="trial counts"):
        compare(learned, wrong)
