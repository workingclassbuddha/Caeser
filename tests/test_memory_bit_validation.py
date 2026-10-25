"""Freeze and externally validate a tiny memory rule without retraining."""
import pytest

from caesar.memory_bit_validation import validate


def test_same_rule_and_seeded_trials_are_repeatable():
    a = validate([89], trials=2, pairs_per_seed=2)
    b = validate([89], trials=2, pairs_per_seed=2)
    assert a == b
    assert a["protocol"] == "caesar_v2_frozen_one_bit_memory_independent_validation_v1"
    assert set(a["summary"]) == {"default_up", "stored_ray_bit"}
    assert a["summary"]["default_up"]["n"] == 16
    assert a["summary"]["stored_ray_bit"]["n"] == 16
    assert 0.0 <= a["cue_accuracy"]["overall"] <= 1.0
    assert len(a["cue_trials"]) == 2
    with pytest.raises(ValueError):
        validate([89,89], trials=1, pairs_per_seed=1)
