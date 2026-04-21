from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .core import CAESARNet, EmbodiedEnv
from .training import (
    TrainingConfig,
    apply_actor_critic_update,
    compute_noise_scale,
    promotion_threshold,
    resolve_device,
    run_training_episode,
    set_seed,
)


@dataclass(slots=True)
class ExperimentCondition:
    name: str
    description: str
    scar_decay_per_episode: float = 0.0
    log: list[dict[str, float | int]] = field(default_factory=list)

    def should_demote(
        self,
        level: int,
        mirror_history: list[float],
        episodes_since_promotion: int,
    ) -> tuple[bool, int]:
        return False, level

    def apply_post_episode_effects(self, agent: CAESARNet) -> None:
        if self.scar_decay_per_episode <= 0.0:
            return
        with torch.no_grad():
            agent.scar.mul_(max(0.0, 1.0 - self.scar_decay_per_episode))

    def log_episode(
        self,
        episode: int,
        mirror: float,
        mirror_mean: float,
        mirror_variance: float,
        success_rate: float,
        scar: float,
        level: int,
        time_at_level: int,
    ) -> None:
        self.log.append(
            {
                "episode": episode,
                "mirror": mirror,
                "mirror_mean": mirror_mean,
                "mirror_variance": mirror_variance,
                "success_rate": success_rate,
                "scar": scar,
                "level": level,
                "time_at_level": time_at_level,
            }
        )


class AdaptiveDemotionCondition(ExperimentCondition):
    def __init__(self) -> None:
        super().__init__(
            name="ADAPTIVE_DEMOTION",
            description="Demote when mirror variance collapses and the agent remains stuck.",
        )

    def should_demote(
        self,
        level: int,
        mirror_history: list[float],
        episodes_since_promotion: int,
    ) -> tuple[bool, int]:
        if len(mirror_history) >= 100 and episodes_since_promotion > 200:
            mirror_variance = float(np.var(mirror_history))
            mirror_mean = float(np.mean(mirror_history))
            if mirror_variance < 0.01 and mirror_mean < 0.25:
                return True, max(0, level - 2)
        return False, level


class ScarDecayCondition(ExperimentCondition):
    def __init__(self) -> None:
        super().__init__(
            name="SCAR_DECAY",
            description="Apply a 0.1% scar decay after each episode to test partial healing.",
            scar_decay_per_episode=0.001,
        )


class HealingDemotionCondition(AdaptiveDemotionCondition):
    def __init__(self) -> None:
        super().__init__()
        self.name = "HEALING_DEMOTION"
        self.description = "Combine adaptive demotion with 0.1% scar decay per episode."
        self.scar_decay_per_episode = 0.001


class OscillationTrackerCondition(ExperimentCondition):
    def __init__(self) -> None:
        super().__init__(
            name="OSCILLATION_TRACKER",
            description="Track freeze/melt transitions to inspect breathing-band recovery.",
        )
        self.freeze_count = 0
        self.melt_count = 0
        self.last_mirror: float | None = None

    def log_episode(
        self,
        episode: int,
        mirror: float,
        mirror_mean: float,
        mirror_variance: float,
        success_rate: float,
        scar: float,
        level: int,
        time_at_level: int,
    ) -> None:
        if self.last_mirror is not None:
            if mirror > self.last_mirror + 0.05:
                self.freeze_count += 1
            elif mirror < self.last_mirror - 0.05:
                self.melt_count += 1
        self.last_mirror = mirror

        self.log.append(
            {
                "episode": episode,
                "mirror": mirror,
                "mirror_mean": mirror_mean,
                "mirror_variance": mirror_variance,
                "success_rate": success_rate,
                "scar": scar,
                "level": level,
                "time_at_level": time_at_level,
                "freeze_count": self.freeze_count,
                "melt_count": self.melt_count,
                "oscillation_frequency": (self.freeze_count + self.melt_count) / max(episode + 1, 1),
            }
        )


def default_conditions() -> list[ExperimentCondition]:
    return [
        ExperimentCondition(
            name="BASELINE",
            description="Original v12.10 dynamics with persistent scar and no demotion.",
        ),
        AdaptiveDemotionCondition(),
        ScarDecayCondition(),
        HealingDemotionCondition(),
        OscillationTrackerCondition(),
    ]


def train_condition(
    condition: ExperimentCondition,
    max_episodes: int = 3_000,
    seed: int = 42,
    device_name: str | None = None,
    log_every: int = 20,
) -> ExperimentCondition:
    set_seed(seed)
    device = resolve_device(device_name)
    training_config = TrainingConfig(episodes=max_episodes, seed=seed, device=str(device), log_every=log_every)

    env = EmbodiedEnv(level=0)
    agent = CAESARNet().to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=training_config.learning_rate)

    success_window: deque[int] = deque(maxlen=training_config.success_window_size)
    mirror_history: deque[float] = deque(maxlen=100)
    level = 0
    time_at_level = 0
    episodes_since_promotion = 0

    print(f"\n{'=' * 80}")
    print(f"TESTING: {condition.name}")
    print(f"Description: {condition.description}")
    print(f"{'=' * 80}\n")

    for episode in range(max_episodes):
        noise_scale = compute_noise_scale(episode, training_config)
        rollout = run_training_episode(env, agent, device, noise_scale)
        apply_actor_critic_update(agent, optimizer, rollout, training_config, device)
        condition.apply_post_episode_effects(agent)

        success_window.append(1 if rollout.success else 0)
        mirror_history.append(float(agent.M_state.item()))
        time_at_level += 1
        episodes_since_promotion += 1

        if episode % log_every == 0:
            success_rate = float(np.mean(success_window)) if success_window else 0.0
            mirror_mean = float(np.mean(mirror_history)) if mirror_history else 0.0
            mirror_variance = float(np.var(mirror_history)) if len(mirror_history) > 1 else 0.0

            condition.log_episode(
                episode=episode,
                mirror=float(agent.M_state.item()),
                mirror_mean=mirror_mean,
                mirror_variance=mirror_variance,
                success_rate=success_rate,
                scar=float(agent.scar.item()),
                level=level,
                time_at_level=time_at_level,
            )

            print(
                "Ep {episode:4d} | M {mirror:.4f} | M_bar {mirror_mean:.3f} | Var(M) {mirror_variance:.4f} | "
                "Scar {scar:.5f} | SR {success_rate:.2f} | Lvl {level:2d} | T@L {time_at_level}".format(
                    episode=episode,
                    mirror=agent.M_state.item(),
                    mirror_mean=mirror_mean,
                    mirror_variance=mirror_variance,
                    scar=agent.scar.item(),
                    success_rate=success_rate,
                    level=level,
                    time_at_level=time_at_level,
                )
            )

            should_demote, new_level = condition.should_demote(
                level,
                list(mirror_history),
                episodes_since_promotion,
            )
            if should_demote and new_level < level:
                print(f"  >>> THERAPEUTIC DEMOTION: Level {level} -> {new_level} <<<")
                print(f"      Reason: Var(M)={mirror_variance:.6f}, Mean(M)={mirror_mean:.4f}")
                level = new_level
                env.level = level
                success_window.clear()
                mirror_history.clear()
                time_at_level = 0
                episodes_since_promotion = 0
                continue

            threshold = promotion_threshold(level, training_config)
            if success_rate >= threshold and agent.M_state.item() > 0.18:
                level += 1
                env.level = level
                success_window.clear()
                time_at_level = 0
                episodes_since_promotion = 0
                print(f"  >>> NUCLEATION SUCCESS: Level {level} <<<")

    return condition


def analyze_oscillation_pattern(log_data: list[dict[str, float | int]]) -> str:
    mirrors = [float(entry["mirror"]) for entry in log_data]
    if len(mirrors) < 10:
        return "Insufficient data"

    # At the default logging cadence, 25 points correspond to roughly the final 500 episodes.
    analysis_window = min(25, len(mirrors))
    final_mirrors = mirrors[-analysis_window:]
    mean_final = float(np.mean(final_mirrors))
    variance_final = float(np.var(final_mirrors))

    oscillations = 0
    for index in range(1, len(final_mirrors)):
        previous_value = final_mirrors[index - 1]
        current_value = final_mirrors[index]
        if (current_value > mean_final and previous_value < mean_final) or (
            current_value < mean_final and previous_value > mean_final
        ):
            oscillations += 1

    oscillation_frequency = oscillations / len(final_mirrors)
    if variance_final < 0.005:
        return (
            "STATIC ICE: M stable at "
            f"{mean_final:.3f}, variance={variance_final:.6f}, osc_freq={oscillation_frequency:.3f}"
        )
    return (
        "BREATHING: M oscillating around "
        f"{mean_final:.3f}, variance={variance_final:.6f}, osc_freq={oscillation_frequency:.3f}"
    )


def compare_conditions(conditions: list[ExperimentCondition]) -> None:
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS")
    print("=" * 80 + "\n")

    for condition in conditions:
        print(f"\n{condition.name}:")
        print(f"  Description: {condition.description}")

        if not condition.log:
            print("  No data logged")
            continue

        final_entries = condition.log[-10:]
        status_window = condition.log[-min(20, len(condition.log)) :]
        final_mirror = float(np.mean([float(entry["mirror"]) for entry in final_entries]))
        final_level = int(final_entries[-1]["level"])
        total_scar = float(final_entries[-1]["scar"])

        print(f"  Final M: {final_mirror:.4f}")
        print(f"  Final Level: {final_level}")
        print(f"  Total Scar: {total_scar:.6f}")
        print(f"  Pattern: {analyze_oscillation_pattern(condition.log)}")

        collapsed = all(float(entry["mirror"]) < 0.20 for entry in status_window)
        print("  STATUS: COLLAPSED" if collapsed else "  STATUS: FUNCTIONAL")

    print("\n" + "=" * 80)


def save_results(
    conditions: list[ExperimentCondition],
    filename: str = "outputs/caesar_condition_results.json",
) -> None:
    destination = Path(filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    results = {
        condition.name: {
            "description": condition.description,
            "log": condition.log,
        }
        for condition in conditions
    }
    destination.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {destination}")


def run_comprehensive_test(
    episodes: int = 3_000,
    seed: int = 42,
    device_name: str | None = None,
    sleep_between: float = 1.0,
    output_path: str = "outputs/caesar_condition_results.json",
) -> list[ExperimentCondition]:
    conditions = default_conditions()
    results: list[ExperimentCondition] = []

    for index, condition in enumerate(conditions, start=1):
        print(f"\n{'#' * 80}")
        print(f"# TEST {index}/{len(conditions)}")
        print(f"{'#' * 80}")
        result = train_condition(
            condition=condition,
            max_episodes=episodes,
            seed=seed + index - 1,
            device_name=device_name,
        )
        results.append(result)
        if sleep_between > 0:
            time.sleep(sleep_between)

    compare_conditions(results)
    save_results(results, filename=output_path)

    print("\n" + "=" * 80)
    print("CONSTRAINT DYNAMICS TEST COMPLETE")
    print("=" * 80)
    print("Recovered systems are evaluated by their final oscillation pattern, not by peak mirror alone.")
    print("=" * 80 + "\n")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CAESAR comparative condition experiments.")
    parser.add_argument("--episodes", type=int, default=3_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--sleep-between", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/caesar_condition_results.json",
        help="Path for the JSON experiment log.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_comprehensive_test(
        episodes=args.episodes,
        seed=args.seed,
        device_name=args.device,
        sleep_between=args.sleep_between,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
