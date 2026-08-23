from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal

from caesar.core import EmbodiedEnv, EnvConfig
from caesar.observer import ReflexiveCAESARNet, measure_observer_order
from caesar.training import EpisodeRollout, TrainingConfig, apply_actor_critic_update, compute_noise_scale, resolve_device, set_seed


def run_variant(name: str, recursive: bool, episodes: int, seed: int, device_name: str | None) -> dict:
    set_seed(seed)
    device = resolve_device(device_name)
    cfg = TrainingConfig(episodes=episodes, seed=seed, device=str(device), log_every=20)
    env = EmbodiedEnv(level=0, config=EnvConfig())
    agent = ReflexiveCAESARNet(recursive=recursive).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate)
    hidden_trace: list[np.ndarray] = []
    self_trace: list[np.ndarray] = []
    metrics_log: list[dict] = []
    success: list[int] = []

    for episode in range(episodes):
        observation = env.reset()
        hidden_state = torch.zeros(1, 1, agent.hidden_size, device=device)
        self_state = torch.zeros(1, 1, agent.hidden_size, device=device)
        rollout = EpisodeRollout([], [], [], False, 0.0, 0)
        self_losses: list[torch.Tensor] = []
        done = False
        while not done:
            obs = torch.as_tensor(observation, dtype=torch.float32, device=device).view(1, 1, -1)
            mean, value, hidden_state, self_state, predicted_self = agent(obs, hidden_state, env.battery, self_state)
            # The self-model predicts the next internal state from the current internal state.
            # Its auxiliary loss is deliberately tiny so task learning remains dominant.
            self_losses.append(torch.mean((predicted_self - hidden_state.detach()) ** 2))
            std = torch.exp(agent.log_std).clamp(1e-3, 1.0) * compute_noise_scale(episode, cfg)
            dist = Normal(mean, std.view(1, 1, -1))
            action = dist.sample()
            observation, reward, done, info = env.step(action.squeeze().detach().cpu().numpy())
            rollout.rewards.append(float(reward))
            rollout.log_probs.append(dist.log_prob(action).sum(dim=-1).reshape(()))
            rollout.values.append(value.reshape(()))
            rollout.total_reward += float(reward)
            rollout.success = bool(info["success"])
            rollout.steps += 1
            hidden_trace.append(hidden_state.detach().cpu().numpy().reshape(-1))
            self_trace.append(predicted_self.detach().cpu().numpy().reshape(-1))

        apply_actor_critic_update(agent, optimizer, rollout, cfg, device)
        # Auxiliary self-model training makes R a learned variable rather than a decorative readout.
        optimizer.zero_grad(set_to_none=True)
        aux_loss = torch.stack(self_losses).mean()
        (0.05 * aux_loss).backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.gradient_clip_norm)
        optimizer.step()
        success.append(int(rollout.success))

        if episode % 20 == 0 and len(hidden_trace) >= 10:
            h = np.asarray(hidden_trace[-500:])
            p = np.asarray(self_trace[-500:])
            metrics = measure_observer_order(h, p)
            row = {"episode": episode, **metrics.as_dict(), "mirror": float(agent.M_state.item()), "scar": float(agent.scar.item()), "success_rate": float(np.mean(success[-50:])), "self_model_loss": float(aux_loss.detach().cpu())}
            metrics_log.append(row)
            print(name, row)

    return {"variant": name, "recursive": recursive, "episodes": episodes, "final": metrics_log[-1] if metrics_log else {}, "history": metrics_log}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the CAESAR observer-transition experiment.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="outputs/observer_transition_results.json")
    args = parser.parse_args()
    results = [
        run_variant("REFLEXIVE", False, args.episodes, args.seed, args.device),
        run_variant("RECURSIVE", True, args.episodes, args.seed + 1, args.device),
    ]
    payload = {
        "hypothesis": "observer-like organization is associated with integration + persistence + closure + self-model irreducibility",
        "note": "Observer order is an operational proxy, not evidence of subjective experience.",
        "results": results,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
