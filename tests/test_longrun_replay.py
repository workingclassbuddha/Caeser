"""Corrected historical-protocol replay smoke tests."""
import json

import pytest

from caesar.longrun_replay import replay


def test_replay_preserves_seed_and_episode_log(tmp_path):
    path = tmp_path / "replication.json"
    result = replay(seeds=[7], episodes=1, output=str(path))
    assert result["episodes"] == 1
    assert result["seeds"] == [7]
    assert result["runs"][0]["seed"] == 7
    assert result["runs"][0]["snapshots"][0]["episode"] == 0
    assert json.loads(path.read_text())["protocol"] == "corrected_caesar_longrun_curriculum_v1"
    assert (tmp_path / "curriculum_seed_7.json").exists()


def test_replay_rejects_duplicate_seeds(tmp_path):
    with pytest.raises(ValueError):
        replay([1, 1], 2, str(tmp_path / "replication.json"))
