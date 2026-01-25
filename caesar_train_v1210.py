import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from collections import deque

# ============================================================================
# 1. THE EMBODIED ENVIRONMENT (Precision Lattice)
# ============================================================================
class EmbodiedEnv:
    def __init__(self, level=0):
        self.level = level
        self.reset()

    def reset(self):
        self.size = 5.0 + (self.level / 40.0)
        self.max_steps = 350 + (self.level * 15) 
        # Adaptive radius: starts at 1.0, floor at 0.40
        self.success_radius = max(0.40, 1.0 - (self.level * 0.05))
        
        self.agent_pos = np.random.uniform(1.0, self.size - 1.0, 2)
        self.goal_pos = np.random.uniform(1.0, self.size - 1.0, 2)
        self.velocity = np.zeros(2)
        self.battery = 100.0
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        delta = self.goal_pos - self.agent_pos
        dist = np.linalg.norm(delta)
        wdir = delta / dist if dist > 1e-6 else np.zeros(2)
        obs = [self.velocity[0], self.velocity[1], wdir[0], wdir[1], 
               dist / self.size, self.battery / 100.0]
        
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        for a in angles:
            ray = np.array([np.cos(a), np.sin(a)])
            d = self.size * 2.0
            if ray[0] > 0: d = min(d, (self.size - self.agent_pos[0]) / ray[0])
            elif ray[0] < 0: d = min(d, -self.agent_pos[0] / ray[0])
            if ray[1] > 0: d = min(d, (self.size - self.agent_pos[1]) / ray[1])
            elif ray[1] < 0: d = min(d, -self.agent_pos[1] / ray[1])
            obs.append(d / (self.size * 2.0))
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.steps += 1
        thrust = np.clip(action, -1.0, 1.0)
        self.velocity = self.velocity * 0.85 + thrust * 0.25
        self.agent_pos += self.velocity * 0.12
        self.agent_pos = np.clip(self.agent_pos, 0.1, self.size - 0.1)
        
        self.battery -= (0.02 + np.linalg.norm(thrust) * 0.15)
        dist = np.linalg.norm(self.agent_pos - self.goal_pos)
        
        time_weight = 1.0 + (self.steps / 150.0)
        reward = -0.3 * dist * time_weight

        success = False
        if dist < self.success_radius:
            reward += 500.0 + (self.battery * 35.0) 
            success = True
            done = True
        elif self.battery <= 0 or self.steps >= self.max_steps:
            reward -= 100.0
            done = True
        else:
            done = False
        return self._get_obs(), reward, done, {'success': success}

# ============================================================================
# 2. CAESAR V12.10 NET
# ============================================================================
class CAESARNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(14, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        self.gru = nn.GRU(128, 128, batch_first=True)
        self.actor = nn.Linear(128, 2)
        self.critic = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.full((2,), -1.0)) 
        
        self.register_buffer('scar', torch.zeros(1))
        self.register_buffer('M_state', torch.tensor(1.0))

    def forward(self, x, h, battery):
        z = self.embed(x)
        t = torch.tensor(max(battery, 0.0) / 100.0, device=x.device)
        l = 1.0 / (1.0 + torch.var(z) + 1e-6)
        M = (l * t) + 1e-6
        self.M_state = 0.99 * self.M_state + 0.01 * M

        if self.M_state < 0.12 and battery < 5.0:
            self.scar += 0.000005 
        
        z_out, h = self.gru(z, h)
        return torch.tanh(self.actor(z_out)), self.critic(z_out), h

# ============================================================================
# 3. TRAINER (Jumpstart Protocol)
# ============================================================================
def train_caesar():
    env = EmbodiedEnv(level=0)
    agent = CAESARNet()
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    success_window = deque(maxlen=50)
    level = 0

    print("--- CAESAR V12.10: NUCLEATION JUMPSTART ---")

    for ep in range(30000):
        obs = env.reset()
        h = torch.zeros(1, 1, 128)
        done, rewards, log_probs, values = False, [], [], []
        
        # Consistent Precision Anchor
        noise_multiplier = max(0.25, 1.2 - (ep / 1200))

        while not done:
            st = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0)
            mu, val, h = agent(st, h, env.battery)
            
            std = torch.exp(agent.log_std).clamp(1e-3, 1.0) * noise_multiplier
            dist_dist = Normal(mu, std)
            action = dist_dist.sample()
            
            next_obs, rew, done, info = env.step(action[0,0].detach().numpy())
            log_probs.append(dist_dist.log_prob(action).sum(-1))
            values.append(val.squeeze())
            rewards.append(rew)
            obs = next_obs

        returns, G = [], 0
        for r in reversed(rewards):
            G = r + 0.98 * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        adv = (returns - torch.stack(values).detach())
        health_factor = 1.0 / (agent.scar.item() + 0.1)
        policy_loss = -(torch.cat(log_probs) * adv * health_factor).mean()
        value_loss = F.mse_loss(torch.stack(values), returns)
        
        optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()

        success_window.append(1 if info.get('success') else 0)

        if ep % 20 == 0:
            sr = np.mean(success_window)
            print(f"Ep {ep:4d} | M {agent.M_state.item():.4f} | Scar {agent.scar.item():.4f} | SR {sr:.2f} | Rad {env.success_radius:.2f} | Lvl {level}")
            
            # --- V12.10 JUMPSTART LOGIC ---
            if level == 0:
                promo_threshold = 0.50  # Lower threshold to break the initial stall
            elif level >= 10:
                promo_threshold = 0.60  # Precision-heavy threshold
            else:
                promo_threshold = 0.70  # Standard curriculum gate
                
            if sr >= promo_threshold and agent.M_state.item() > 0.18:
                level += 1
                env.level = level
                print(f"  >>> NUCLEATION SUCCESS: Level {level} <<<")
                success_window.clear()

if __name__ == "__main__":
    train_caesar()


