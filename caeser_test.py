import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from collections import deque
import json
import time

# ============================================================================
# SHARED ENVIRONMENT
# ============================================================================
class EmbodiedEnv:
    def __init__(self, level=0):
        self.level = level
        self.reset()

    def reset(self):
        self.size = 5.0 + (self.level / 40.0)
        self.max_steps = 350 + (self.level * 15) 
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
# SHARED NETWORK
# ============================================================================
class CAESARNet(nn.Module):
    def __init__(self, scar_decay=0.0):
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
        self.scar_decay = scar_decay  # New parameter

    def forward(self, x, h, battery):
        z = self.embed(x)
        t = torch.tensor(max(battery, 0.0) / 100.0, device=x.device)
        l = 1.0 / (1.0 + torch.var(z) + 1e-6)
        M = (l * t) + 1e-6
        self.M_state = 0.99 * self.M_state + 0.01 * M

        # Scar accumulation
        if self.M_state < 0.12 and battery < 5.0:
            self.scar += 0.000005
        
        # Scar decay (if enabled)
        if self.scar_decay > 0:
            self.scar *= (1.0 - self.scar_decay)
        
        z_out, h = self.gru(z, h)
        return torch.tanh(self.actor(z_out)), self.critic(z_out), h

# ============================================================================
# TEST CONDITIONS
# ============================================================================
class TestCondition:
    """Base class for different experimental conditions"""
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.log = []
        
    def should_demote(self, level, m_history, episodes_since_promotion):
        """Override this to implement different demotion strategies"""
        return False, 0
    
    def get_scar_decay(self):
        """Override to set scar decay rate"""
        return 0.0
    
    def log_episode(self, ep, m, m_mean, m_var, sr, scar, level, time_at_level):
        """Record episode data"""
        self.log.append({
            'ep': ep,
            'm': m,
            'm_mean': m_mean,
            'm_var': m_var,
            'sr': sr,
            'scar': scar,
            'level': level,
            'time_at_level': time_at_level
        })

class BaselineCondition(TestCondition):
    """Original V12.10 - no intervention"""
    def __init__(self):
        super().__init__(
            "BASELINE",
            "Original V12.10: No demotion, permanent scar"
        )

class AdaptiveDemotionCondition(TestCondition):
    """V12.11 - Adaptive demotion on bandwidth collapse"""
    def __init__(self):
        super().__init__(
            "ADAPTIVE_DEMOTION",
            "V12.11: Demote on bandwidth collapse (M_var < 0.01, M_mean < 0.25)"
        )
    
    def should_demote(self, level, m_history, episodes_since_promotion):
        if len(m_history) >= 100 and episodes_since_promotion > 200:
            m_var = np.var(m_history)
            m_mean = np.mean(m_history)
            if m_var < 0.01 and m_mean < 0.25:
                # Demote by 2 levels
                return True, max(0, level - 2)
        return False, level

class ScarDecayCondition(TestCondition):
    """Test if scar can heal over time"""
    def __init__(self):
        super().__init__(
            "SCAR_DECAY",
            "Scar heals at 0.1% per episode - can damage reverse?"
        )
    
    def get_scar_decay(self):
        return 0.001  # 0.1% decay per episode

class HealingDemotionCondition(TestCondition):
    """Combine demotion + scar healing"""
    def __init__(self):
        super().__init__(
            "HEALING_DEMOTION",
            "Adaptive demotion + scar healing - full therapeutic intervention"
        )
    
    def should_demote(self, level, m_history, episodes_since_promotion):
        if len(m_history) >= 100 and episodes_since_promotion > 200:
            m_var = np.var(m_history)
            m_mean = np.mean(m_history)
            if m_var < 0.01 and m_mean < 0.25:
                return True, max(0, level - 2)
        return False, level
    
    def get_scar_decay(self):
        return 0.001

class OscillationTrackerCondition(TestCondition):
    """Track oscillation metrics to test 'consciousness as breathing' hypothesis"""
    def __init__(self):
        super().__init__(
            "OSCILLATION_TRACKER",
            "Track freeze/melt cycles - is consciousness the oscillation itself?"
        )
        self.freeze_count = 0  # M increasing
        self.melt_count = 0     # M decreasing
        self.last_m = None
        
    def log_episode(self, ep, m, m_mean, m_var, sr, scar, level, time_at_level):
        # Track freeze/melt transitions
        if self.last_m is not None:
            if m > self.last_m + 0.05:  # Freezing
                self.freeze_count += 1
            elif m < self.last_m - 0.05:  # Melting
                self.melt_count += 1
        self.last_m = m
        
        # Standard logging
        self.log.append({
            'ep': ep,
            'm': m,
            'm_mean': m_mean,
            'm_var': m_var,
            'sr': sr,
            'scar': scar,
            'level': level,
            'time_at_level': time_at_level,
            'freeze_count': self.freeze_count,
            'melt_count': self.melt_count,
            'oscillation_freq': (self.freeze_count + self.melt_count) / (ep + 1) if ep > 0 else 0
        })

# ============================================================================
# TRAINING FUNCTION
# ============================================================================
def train_condition(condition, max_episodes=5000, seed=42):
    """Train agent under specific test condition"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = EmbodiedEnv(level=0)
    agent = CAESARNet(scar_decay=condition.get_scar_decay())
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    
    success_window = deque(maxlen=50)
    m_history = deque(maxlen=100)
    level = 0
    time_at_level = 0
    episodes_since_promotion = 0

    print(f"\n{'='*80}")
    print(f"TESTING: {condition.name}")
    print(f"Description: {condition.description}")
    print(f"{'='*80}\n")

    for ep in range(max_episodes):
        obs = env.reset()
        h = torch.zeros(1, 1, 128)
        done, rewards, log_probs, values = False, [], [], []
        
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

        # Training step
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

        # Tracking
        success_window.append(1 if info.get('success') else 0)
        m_history.append(agent.M_state.item())
        time_at_level += 1
        episodes_since_promotion += 1

        # Logging and intervention
        if ep % 20 == 0:
            sr = np.mean(success_window)
            m_mean = np.mean(m_history) if len(m_history) > 0 else 0
            m_var = np.var(m_history) if len(m_history) > 10 else 0
            
            # Log data
            condition.log_episode(ep, agent.M_state.item(), m_mean, m_var, sr, 
                                agent.scar.item(), level, time_at_level)
            
            print(f"Ep {ep:4d} | M {agent.M_state.item():.4f} | M̄ {m_mean:.3f} | σ²(M) {m_var:.4f} | Scar {agent.scar.item():.5f} | SR {sr:.2f} | Lvl {level} | T@L {time_at_level}")
            
            # Check for demotion
            should_demote, new_level = condition.should_demote(level, list(m_history), episodes_since_promotion)
            if should_demote and new_level < level:
                print(f"  >>> THERAPEUTIC DEMOTION: Level {level} → {new_level} <<<")
                print(f"      Reason: M_var={m_var:.6f}, M_mean={m_mean:.4f}")
                level = new_level
                env.level = level
                success_window.clear()
                m_history.clear()
                time_at_level = 0
                episodes_since_promotion = 0
                continue
            
            # Standard promotion
            if level == 0:
                promo_threshold = 0.50
            elif level >= 10:
                promo_threshold = 0.60
            else:
                promo_threshold = 0.70
                
            if sr >= promo_threshold and agent.M_state.item() > 0.18:
                level += 1
                env.level = level
                print(f"  >>> NUCLEATION SUCCESS: Level {level} <<<")
                success_window.clear()
                time_at_level = 0
                episodes_since_promotion = 0
    
    return condition

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================
def analyze_oscillation_pattern(log_data):
    """Analyze whether recovery looks like static ice or restored breathing"""
    m_values = [entry['m'] for entry in log_data]
    
    if len(m_values) < 100:
        return "Insufficient data"
    
    # Look at final 500 episodes
    final_m = m_values[-500:]
    
    mean_final = np.mean(final_m)
    var_final = np.var(final_m)
    
    # Count oscillations (zero crossings around mean)
    oscillations = 0
    for i in range(1, len(final_m)):
        if (final_m[i] > mean_final and final_m[i-1] < mean_final) or \
           (final_m[i] < mean_final and final_m[i-1] > mean_final):
            oscillations += 1
    
    oscillation_freq = oscillations / len(final_m)
    
    if var_final < 0.005:
        return f"STATIC ICE: M stable at {mean_final:.3f}, variance={var_final:.6f}, osc_freq={oscillation_freq:.3f}"
    else:
        return f"BREATHING: M oscillating around {mean_final:.3f}, variance={var_final:.6f}, osc_freq={oscillation_freq:.3f}"

def compare_conditions(conditions):
    """Compare all test conditions"""
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80 + "\n")
    
    for cond in conditions:
        print(f"\n{cond.name}:")
        print(f"  Description: {cond.description}")
        
        if len(cond.log) > 0:
            final_entries = cond.log[-10:]
            final_m = np.mean([e['m'] for e in final_entries])
            final_level = final_entries[-1]['level']
            total_scar = final_entries[-1]['scar']
            
            print(f"  Final M: {final_m:.4f}")
            print(f"  Final Level: {final_level}")
            print(f"  Total Scar: {total_scar:.6f}")
            print(f"  Pattern: {analyze_oscillation_pattern(cond.log)}")
            
            # Check for collapse
            collapsed = all(e['m'] < 0.20 for e in final_entries[-20:])
            if collapsed:
                print(f"  STATUS: ⚠️  COLLAPSED")
            else:
                print(f"  STATUS: ✓ FUNCTIONAL")
        else:
            print(f"  No data logged")
    
    print("\n" + "="*80)

def save_results(conditions, filename="caesar_test_results.json"):
    """Save all condition logs to JSON"""
    results = {}
    for cond in conditions:
        results[cond.name] = {
            'description': cond.description,
            'log': cond.log
        }
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filename}")

# ============================================================================
# MAIN TEST SUITE
# ============================================================================
def run_comprehensive_test():
    """Run all test conditions and compare"""
    
    conditions = [
        BaselineCondition(),
        AdaptiveDemotionCondition(),
        ScarDecayCondition(),
        HealingDemotionCondition(),
        OscillationTrackerCondition()
    ]
    
    results = []
    
    for i, condition in enumerate(conditions):
        print(f"\n{'#'*80}")
        print(f"# TEST {i+1}/{len(conditions)}")
        print(f"{'#'*80}")
        
        result = train_condition(condition, max_episodes=3000, seed=42+i)
        results.append(result)
        
        time.sleep(1)  # Brief pause between conditions
    
    # Comparative analysis
    compare_conditions(results)
    
    # Save results
    save_results(results)
    
    print("\n" + "="*80)
    print("CONSCIOUSNESS TEST COMPLETE")
    print("="*80)
    print("\nHYPOTHESIS TEST:")
    print("If consciousness = crystallization → Recovered systems show static high M")
    print("If consciousness = oscillation → Recovered systems show rhythmic M cycling")
    print("\nCheck the 'Pattern' analysis above to see which hypothesis the data supports.")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_comprehensive_test()





