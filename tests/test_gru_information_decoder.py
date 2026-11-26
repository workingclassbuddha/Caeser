"""Information, not behaviour: validate strictly held-out GRU decoders."""
import numpy as np
import pytest
import torch

from caesar.core import CAESARNet
from caesar.gru_information_decoder import (
    FEATURES, TRAIN_WORLD_SEEDS, TEST_WORLD_SEEDS,
    feature_rows, ridge_probe, one_bit_accuracy, probe_one_model,
)


def test_train_test_worlds_disjoint_and_both_gap_sides_balanced():
    assert not set(TRAIN_WORLD_SEEDS).intersection(TEST_WORLD_SEEDS)
    torch.manual_seed(11)
    net = CAESARNet()
    train = feature_rows(net, TRAIN_WORLD_SEEDS[:1], 4)
    test = feature_rows(net, TEST_WORLD_SEEDS[:1], 4)
    assert train and test
    assert len(train) % 2 == 0 and len(test) % 2 == 0
    for rows in (train, test):
        for a, b in zip(rows[0::2], rows[1::2]):
            assert a["label"] == "top" and b["label"] == "bottom"
            assert a["world_seed"] == b["world_seed"]
            assert a["trial"] == b["trial"]
            np.testing.assert_allclose(a["current_observation"], b["current_observation"], atol=1e-6)
            np.testing.assert_allclose(a["gru_reset"], b["gru_reset"], atol=1e-6)


def test_linear_probe_does_not_expose_labels_across_seed_splits():
    torch.manual_seed(11)
    agent = CAESARNet()
    train = feature_rows(agent, TRAIN_WORLD_SEEDS[:1], 4)
    test = feature_rows(agent, TEST_WORLD_SEEDS[:1], 4)
    for feature in FEATURES:
        result = ridge_probe(train, test, feature)
        assert result["train_examples"] == len(train)
        assert result["test_examples"] == len(test)
        assert 0 <= result["test_accuracy"] <= 1
        assert len(result["test_predictions"]) == len(test)
    # Identical final observations with opposing gap labels have
    # necessarily 50% balanced label accuracy under a fixed classifier.
    final = ridge_probe(train, test, "current_observation")
    reset = ridge_probe(train, test, "gru_reset")
    assert final["test_accuracy"] == 0.5
    assert reset["test_accuracy"] == 0.5
    with pytest.raises(ValueError, match="overlap"):
        ridge_probe(train, train, "gru_preserved")
    with pytest.raises(ValueError):
        ridge_probe([], test, "gru_preserved")


def test_training_zero_episodes_leaves_model_identical_and_reports_one_bit():
    result = probe_one_model(seed=11, episodes=0, train_trials=3, test_trials=3)
    assert result["training_successes"] == 0
    before = result["stages"]["untrained"]
    after = result["stages"]["trained"]
    for feature in FEATURES:
        assert (
            before["probe_results"][feature]["test_accuracy"]
            == after["probe_results"][feature]["test_accuracy"]
        )
    assert before["fixed_one_bit_test"] == one_bit_accuracy(before["test_features"])


def test_shuffled_training_labels_are_not_hidden_ground_truth_features():
    torch.manual_seed(23)
    net = CAESARNet()
    train = feature_rows(net, TRAIN_WORLD_SEEDS[:1], 3)
    test = feature_rows(net, TEST_WORLD_SEEDS[:1], 3)
    a = ridge_probe(train, test, "gru_preserved", shuffle_labels=True)
    b = ridge_probe(train, test, "gru_preserved", shuffle_labels=True)
    assert a == b
    assert a["shuffle_training_labels"]
