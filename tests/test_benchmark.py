from caesar.benchmark import metrics, paired_summary


def test_metrics_reports_task_outcomes():
    logs = [
        {"episode": 0, "level": 0, "success_rate": 0.2, "scar": 0.0},
        {"episode": 20, "level": 1, "success_rate": 0.6, "scar": 0.01},
    ]
    result = metrics(logs)
    assert result["last_success_rate"] == 0.6
    assert result["final_level"] == 1
    assert result["first_level_gain_episode"] == 20
    assert result["max_scar"] == 0.01


def test_paired_difference_direction_and_scar_detection():
    sample = [
        {"baseline": {"last_success_rate": 0.8, "mean_success_rate": 0.6,
                       "final_level": 2, "max_scar": 0.01},
         "no_scar": {"last_success_rate": 0.4, "mean_success_rate": 0.5,
                     "final_level": 1, "max_scar": 0.0}}
    ]
    summary = paired_summary(sample)
    assert summary["mean_paired_differences"]["last_success_rate"] == 0.4
    assert summary["mean_paired_differences"]["final_level"] == 1.0
    assert summary["baseline_any_scar"] is True


def test_empty_run_is_not_claimed_as_success():
    assert metrics([])["last_success_rate"] is None
