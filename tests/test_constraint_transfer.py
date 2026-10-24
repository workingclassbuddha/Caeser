"""Structural controls for the preregistered space/time/energy comparison."""
import torch

from caesar.constraint_transfer import ARMS, BASE, arm_config, evaluate, run_study
from caesar.core import CAESARNet


def test_all_eight_factorial_conditions_are_present():
    assert len(ARMS) == 8
    assert len(set(ARMS)) == 8
    assert set(ARMS) == {
        "none", "space", "time", "energy",
        "space+time", "space+energy", "time+energy", "space+time+energy",
    }


def test_each_factor_changes_exactly_one_declared_limit():
    none = arm_config("none")
    space = arm_config("space")
    time = arm_config("time")
    energy = arm_config("energy")

    assert none == BASE
    def changed(a, b):
        return {field for field in a.__dataclass_fields__
                if getattr(a, field) != getattr(b, field)}
    assert changed(none, space) == {"initial_success_radius", "success_radius_floor"}
    assert changed(none, time) == {"base_max_steps"}
    assert changed(none, energy) == {"thrust_battery_cost"}
    combined = arm_config("space+time+energy")
    assert changed(none, combined) == {
        "initial_success_radius", "success_radius_floor",
        "base_max_steps", "thrust_battery_cost",
    }
    assert combined.min_spawn_distance == none.min_spawn_distance


def test_evaluation_preserves_model_state_and_is_repeatable():
    torch.set_num_threads(1)
    torch.manual_seed(89)
    model = CAESARNet()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    a = evaluate(model, "time", seed=7, trials=2)
    b = evaluate(model, "time", seed=7, trials=2)
    assert a == b
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key]), key


def test_full_cross_condition_report_is_reproducible_and_has_training_log():
    one = run_study([17], episodes=1, eval_trials=1, arms=["none"])
    two = run_study([17], episodes=1, eval_trials=1, arms=["none"])
    assert one["protocol"] == "space_time_energy_factorial_v1"
    assert one["summary"] == two["summary"]
    assert len(one["runs"][0]["log"]) == 1
    assert set(one["runs"][0]["end_eval"]) == set(ARMS)
    assert set(one["runs"][0]["start_eval"]) == set(ARMS)
    assert one["runs"][0]["train_config"] == {
        field: getattr(BASE, field) for field in BASE.__dataclass_fields__
    }
