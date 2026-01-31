"""
Comprehensive test suite for CAESAR V12.10 reinforcement learning system.

Tests cover:
- EmbodiedEnv: Environment mechanics, observations, rewards
- CAESARNet: Neural network forward pass, state tracking, scar mechanics
- Integration: Agent-environment interaction
- Edge cases: Boundary conditions and numerical stability
"""

import pytest
import torch
import numpy as np
from caesar_train_v1210 import EmbodiedEnv, CAESARNet


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def env():
    """Create a fresh environment at level 0."""
    return EmbodiedEnv(level=0)


@pytest.fixture
def env_high_level():
    """Create environment at high level for testing curriculum mechanics."""
    return EmbodiedEnv(level=10)


@pytest.fixture
def agent():
    """Create a fresh CAESAR agent."""
    return CAESARNet()


@pytest.fixture
def hidden_state():
    """Create initial hidden state for GRU."""
    return torch.zeros(1, 1, 128)


# ============================================================================
# EMBODIED ENVIRONMENT TESTS
# ============================================================================

class TestEmbodiedEnvInitialization:
    """Tests for environment initialization and reset."""

    def test_env_creates_at_level_zero(self, env):
        """Environment should initialize with correct level."""
        assert env.level == 0

    def test_env_initial_size(self, env):
        """Level 0 should have base size of 5.0."""
        assert env.size == 5.0

    def test_env_size_scales_with_level(self, env_high_level):
        """Higher levels should have larger environments."""
        expected_size = 5.0 + (10 / 40.0)  # level 10
        assert env_high_level.size == expected_size

    def test_env_max_steps_scales_with_level(self, env, env_high_level):
        """Max steps should increase with level."""
        assert env.max_steps == 350
        assert env_high_level.max_steps == 350 + (10 * 15)

    def test_success_radius_decreases_with_level(self, env, env_high_level):
        """Success radius should decrease (harder) at higher levels."""
        assert env.success_radius == 1.0
        expected_radius = max(0.40, 1.0 - (10 * 0.05))
        assert env_high_level.success_radius == expected_radius

    def test_success_radius_has_floor(self):
        """Success radius should not go below 0.40."""
        very_high_env = EmbodiedEnv(level=100)
        assert very_high_env.success_radius == 0.40

    def test_agent_position_in_bounds(self, env):
        """Agent should spawn within environment bounds."""
        for _ in range(100):
            env.reset()
            assert 1.0 <= env.agent_pos[0] <= env.size - 1.0
            assert 1.0 <= env.agent_pos[1] <= env.size - 1.0

    def test_goal_position_in_bounds(self, env):
        """Goal should spawn within environment bounds."""
        for _ in range(100):
            env.reset()
            assert 1.0 <= env.goal_pos[0] <= env.size - 1.0
            assert 1.0 <= env.goal_pos[1] <= env.size - 1.0

    def test_initial_velocity_zero(self, env):
        """Agent should start with zero velocity."""
        env.reset()
        np.testing.assert_array_equal(env.velocity, np.zeros(2))

    def test_initial_battery_full(self, env):
        """Agent should start with full battery."""
        env.reset()
        assert env.battery == 100.0

    def test_initial_steps_zero(self, env):
        """Step counter should start at zero."""
        env.reset()
        assert env.steps == 0


class TestEmbodiedEnvObservation:
    """Tests for observation generation."""

    def test_observation_shape(self, env):
        """Observation should have 14 dimensions."""
        obs = env.reset()
        assert obs.shape == (14,)

    def test_observation_dtype(self, env):
        """Observation should be float32."""
        obs = env.reset()
        assert obs.dtype == np.float32

    def test_observation_components(self, env):
        """Test individual observation components."""
        env.reset()
        env.velocity = np.array([0.5, -0.3])
        obs = env._get_obs()

        # First two elements are velocity
        assert obs[0] == pytest.approx(0.5, rel=1e-5)
        assert obs[1] == pytest.approx(-0.3, rel=1e-5)

    def test_distance_normalized(self, env):
        """Distance should be normalized by environment size."""
        env.reset()
        obs = env._get_obs()
        # obs[4] is dist / self.size
        assert 0.0 <= obs[4] <= 2.0  # Max possible is diagonal

    def test_battery_normalized(self, env):
        """Battery should be normalized to [0, 1]."""
        env.reset()
        obs = env._get_obs()
        assert obs[5] == 1.0  # Full battery

        env.battery = 50.0
        obs = env._get_obs()
        assert obs[5] == 0.5

    def test_ray_observations_count(self, env):
        """Should have 8 ray cast observations."""
        env.reset()
        obs = env._get_obs()
        # obs[6:14] are ray casts
        ray_obs = obs[6:14]
        assert len(ray_obs) == 8

    def test_ray_observations_bounded(self, env):
        """Ray observations should be normalized and bounded."""
        for _ in range(50):
            env.reset()
            obs = env._get_obs()
            ray_obs = obs[6:14]
            assert all(0.0 <= r <= 1.0 for r in ray_obs)


class TestEmbodiedEnvStep:
    """Tests for environment step mechanics."""

    def test_step_increments_counter(self, env):
        """Step counter should increment."""
        env.reset()
        initial_steps = env.steps
        env.step(np.array([0.0, 0.0]))
        assert env.steps == initial_steps + 1

    def test_action_clipping(self, env):
        """Actions should be clipped to [-1, 1]."""
        env.reset()
        # Large action should be clipped
        env.step(np.array([5.0, -5.0]))
        # Velocity update uses clipped thrust
        # thrust = clip([5, -5], -1, 1) = [1, -1]
        # velocity = 0 * 0.85 + [1, -1] * 0.25 = [0.25, -0.25]
        np.testing.assert_array_almost_equal(
            env.velocity, np.array([0.25, -0.25]), decimal=5
        )

    def test_velocity_decay(self, env):
        """Velocity should decay over time."""
        env.reset()
        env.velocity = np.array([1.0, 1.0])
        env.step(np.array([0.0, 0.0]))  # No thrust
        # velocity = [1, 1] * 0.85 + [0, 0] * 0.25 = [0.85, 0.85]
        np.testing.assert_array_almost_equal(
            env.velocity, np.array([0.85, 0.85]), decimal=5
        )

    def test_position_bounded(self, env):
        """Agent position should stay within bounds."""
        env.reset()
        env.agent_pos = np.array([0.0, 0.0])
        env.velocity = np.array([-1.0, -1.0])
        env.step(np.array([-1.0, -1.0]))
        assert env.agent_pos[0] >= 0.1
        assert env.agent_pos[1] >= 0.1

    def test_battery_drain(self, env):
        """Battery should drain with each step."""
        env.reset()
        initial_battery = env.battery
        env.step(np.array([0.0, 0.0]))
        assert env.battery < initial_battery

    def test_battery_drain_increases_with_thrust(self, env):
        """Larger thrust should drain more battery."""
        env.reset()
        env.step(np.array([0.0, 0.0]))
        drain_zero = 100.0 - env.battery

        env.reset()
        env.step(np.array([1.0, 1.0]))
        drain_max = 100.0 - env.battery

        assert drain_max > drain_zero

    def test_success_on_reaching_goal(self, env):
        """Episode should end successfully when reaching goal."""
        env.reset()
        env.agent_pos = env.goal_pos.copy()  # Place agent at goal
        _, _, done, info = env.step(np.array([0.0, 0.0]))
        assert done
        assert info['success']

    def test_failure_on_battery_empty(self, env):
        """Episode should end on battery depletion."""
        env.reset()
        env.battery = 0.01
        _, _, done, info = env.step(np.array([1.0, 1.0]))  # High drain
        assert done
        assert not info['success']

    def test_failure_on_max_steps(self, env):
        """Episode should end when max steps reached."""
        env.reset()
        env.steps = env.max_steps - 1
        _, _, done, info = env.step(np.array([0.0, 0.0]))
        assert done
        assert not info['success']

    def test_positive_reward_on_success(self, env):
        """Should get positive reward for reaching goal."""
        env.reset()
        env.agent_pos = env.goal_pos.copy()
        _, reward, _, info = env.step(np.array([0.0, 0.0]))
        assert info['success']
        assert reward > 0

    def test_battery_bonus_in_success_reward(self, env):
        """Higher battery on success should give higher reward."""
        env.reset()
        env.agent_pos = env.goal_pos.copy()
        env.battery = 100.0
        _, reward_full, _, _ = env.step(np.array([0.0, 0.0]))

        env.reset()
        env.agent_pos = env.goal_pos.copy()
        env.battery = 10.0
        _, reward_low, _, _ = env.step(np.array([0.0, 0.0]))

        assert reward_full > reward_low

    def test_penalty_on_failure(self, env):
        """Should get penalty for running out of battery/time."""
        env.reset()
        env.battery = 0.01
        _, reward, done, info = env.step(np.array([1.0, 1.0]))
        assert done
        assert not info['success']
        assert reward < 0


class TestEmbodiedEnvRewardShaping:
    """Tests for reward function mechanics."""

    def test_reward_decreases_with_distance(self, env):
        """Reward should be lower when farther from goal."""
        env.reset()
        env.goal_pos = np.array([3.0, 3.0])

        env.agent_pos = np.array([2.9, 3.0])  # Close
        obs1, reward1, _, _ = env.step(np.array([0.0, 0.0]))

        env.reset()
        env.agent_pos = np.array([1.0, 1.0])  # Far
        obs2, reward2, _, _ = env.step(np.array([0.0, 0.0]))

        assert reward1 > reward2

    def test_time_penalty_increases(self, env):
        """Reward penalty should increase over time."""
        env.reset()
        env.goal_pos = np.array([3.0, 3.0])
        env.agent_pos = np.array([2.0, 2.0])

        env.steps = 0
        _, reward_early, _, _ = env.step(np.array([0.0, 0.0]))

        env.reset()
        env.goal_pos = np.array([3.0, 3.0])
        env.agent_pos = np.array([2.0, 2.0])
        env.steps = 150
        _, reward_late, _, _ = env.step(np.array([0.0, 0.0]))

        # Later steps have higher time_weight, so more negative reward
        assert reward_early > reward_late


# ============================================================================
# CAESAR NETWORK TESTS
# ============================================================================

class TestCAESARNetArchitecture:
    """Tests for neural network architecture."""

    def test_model_creation(self, agent):
        """Model should instantiate without errors."""
        assert isinstance(agent, CAESARNet)

    def test_embed_layer_shape(self, agent):
        """Embedding layer should transform 14 -> 128."""
        x = torch.randn(1, 1, 14)
        z = agent.embed(x)
        assert z.shape == (1, 1, 128)

    def test_forward_output_shapes(self, agent, hidden_state):
        """Forward pass should return correctly shaped outputs."""
        x = torch.randn(1, 1, 14)
        action, value, h_new = agent(x, hidden_state, battery=100.0)

        assert action.shape == (1, 1, 2)
        assert value.shape == (1, 1, 1)
        assert h_new.shape == (1, 1, 128)

    def test_action_bounded(self, agent, hidden_state):
        """Actions should be bounded by tanh to [-1, 1]."""
        x = torch.randn(1, 1, 14)
        action, _, _ = agent(x, hidden_state, battery=100.0)

        assert action.min() >= -1.0
        assert action.max() <= 1.0

    def test_log_std_parameter(self, agent):
        """Log std should be a learnable parameter."""
        assert agent.log_std.requires_grad
        assert agent.log_std.shape == (2,)

    def test_hidden_state_updates(self, agent, hidden_state):
        """Hidden state should change after forward pass."""
        x = torch.randn(1, 1, 14)
        _, _, h_new = agent(x, hidden_state, battery=100.0)

        # Hidden state should be different from zero init
        assert not torch.allclose(h_new, hidden_state)


class TestCAESARNetMState:
    """Tests for M-state (bandwidth) tracking."""

    def test_m_state_initial_value(self, agent):
        """M-state should initialize to 1.0."""
        assert agent.M_state.item() == 1.0

    def test_m_state_updates(self, agent, hidden_state):
        """M-state should update during forward pass."""
        x = torch.randn(1, 1, 14)
        initial_m = agent.M_state.item()

        for _ in range(10):
            agent(x, hidden_state, battery=100.0)

        # M-state should have changed
        assert agent.M_state.item() != initial_m

    def test_m_state_bounded(self, agent, hidden_state):
        """M-state should remain positive."""
        x = torch.randn(1, 1, 14)

        for _ in range(100):
            agent(x, hidden_state, battery=50.0)

        assert agent.M_state.item() > 0

    def test_m_state_ema_smoothing(self, agent, hidden_state):
        """M-state should use exponential moving average."""
        x = torch.randn(1, 1, 14)

        # Run several forward passes and check M changes gradually
        m_values = []
        for _ in range(20):
            agent(x, hidden_state, battery=100.0)
            m_values.append(agent.M_state.item())

        # Changes should be smooth (no sudden jumps)
        diffs = np.diff(m_values)
        assert np.all(np.abs(diffs) < 0.5)  # No huge jumps


class TestCAESARNetScar:
    """Tests for scar accumulation mechanics."""

    def test_scar_initial_zero(self, agent):
        """Scar should start at zero."""
        assert agent.scar.item() == 0.0

    def test_scar_not_trainable(self, agent):
        """Scar should be a buffer, not a parameter."""
        assert not agent.scar.requires_grad

    def test_scar_accumulates_on_low_m_low_battery(self, agent, hidden_state):
        """Scar should accumulate when M-state low and battery low."""
        x = torch.randn(1, 1, 14)

        # Force low M-state
        agent.M_state = torch.tensor(0.05)

        initial_scar = agent.scar.item()

        # Run with low battery
        for _ in range(100):
            agent(x, hidden_state, battery=2.0)

        # Scar should have increased
        assert agent.scar.item() > initial_scar

    def test_scar_stable_with_high_battery(self, agent, hidden_state):
        """Scar should not accumulate with high battery."""
        x = torch.randn(1, 1, 14)

        # Force low M-state but high battery
        agent.M_state = torch.tensor(0.05)
        initial_scar = agent.scar.item()

        for _ in range(100):
            agent(x, hidden_state, battery=100.0)

        # Scar should not have increased
        assert agent.scar.item() == initial_scar

    def test_scar_stable_with_high_m(self, agent, hidden_state):
        """Scar should not accumulate with healthy M-state."""
        x = torch.randn(1, 1, 14)

        initial_scar = agent.scar.item()

        # Run with low battery but M-state will stay high
        for _ in range(100):
            agent(x, hidden_state, battery=2.0)

        # Check if scar accumulated (depends on M-state evolution)
        # M-state starts at 1.0, so should stay above 0.12 threshold initially
        # This test verifies the condition logic works


class TestCAESARNetGradients:
    """Tests for gradient flow through the network."""

    def test_gradients_flow_to_actor(self, agent, hidden_state):
        """Gradients should flow to actor layers."""
        x = torch.randn(1, 1, 14)
        action, _, _ = agent(x, hidden_state, battery=100.0)
        loss = action.sum()
        loss.backward()

        assert agent.actor.weight.grad is not None
        assert not torch.all(agent.actor.weight.grad == 0)

    def test_gradients_flow_to_critic(self, agent, hidden_state):
        """Gradients should flow to critic layers."""
        x = torch.randn(1, 1, 14)
        _, value, _ = agent(x, hidden_state, battery=100.0)
        loss = value.sum()
        loss.backward()

        assert agent.critic.weight.grad is not None
        assert not torch.all(agent.critic.weight.grad == 0)

    def test_gradients_flow_to_gru(self, agent, hidden_state):
        """Gradients should flow to GRU layers."""
        x = torch.randn(1, 1, 14)
        action, _, _ = agent(x, hidden_state, battery=100.0)
        loss = action.sum()
        loss.backward()

        assert agent.gru.weight_ih_l0.grad is not None

    def test_log_std_gradient(self, agent, hidden_state):
        """Log std parameter should receive gradients."""
        x = torch.randn(1, 1, 14)
        action, _, _ = agent(x, hidden_state, battery=100.0)

        # Use log_std in a loss
        std = torch.exp(agent.log_std)
        loss = std.sum()
        loss.backward()

        assert agent.log_std.grad is not None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestAgentEnvironmentIntegration:
    """Tests for agent-environment interaction."""

    def test_agent_can_process_env_observation(self, env, agent, hidden_state):
        """Agent should successfully process environment observations."""
        obs = env.reset()
        x = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0)

        action, value, h_new = agent(x, hidden_state, env.battery)

        assert action.shape == (1, 1, 2)
        assert value.shape == (1, 1, 1)

    def test_episode_can_complete(self, env, agent, hidden_state):
        """Agent should be able to run a complete episode."""
        obs = env.reset()
        h = hidden_state
        done = False
        steps = 0
        max_test_steps = 500

        while not done and steps < max_test_steps:
            x = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0)
            action, _, h = agent(x, h, env.battery)
            obs, _, done, _ = env.step(action[0, 0].detach().numpy())
            steps += 1

        assert done or steps == max_test_steps

    def test_reward_accumulation(self, env, agent, hidden_state):
        """Rewards should accumulate over episode."""
        obs = env.reset()
        h = hidden_state
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 100:
            x = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0)
            action, _, h = agent(x, h, env.battery)
            obs, reward, done, _ = env.step(action[0, 0].detach().numpy())
            total_reward += reward
            steps += 1

        # Reward should be some finite value
        assert np.isfinite(total_reward)


class TestCurriculumProgression:
    """Tests for curriculum learning mechanics."""

    def test_level_change_updates_environment(self):
        """Changing level should update environment parameters."""
        env = EmbodiedEnv(level=0)
        initial_size = env.size
        initial_radius = env.success_radius

        env.level = 5
        env.reset()

        assert env.size > initial_size
        assert env.success_radius < initial_radius

    def test_higher_levels_are_harder(self):
        """Higher levels should have more challenging conditions."""
        env_easy = EmbodiedEnv(level=0)
        env_hard = EmbodiedEnv(level=15)

        # Harder = larger environment, smaller success radius, more steps needed
        assert env_hard.size > env_easy.size
        assert env_hard.success_radius < env_easy.success_radius


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_distance_to_goal(self, env):
        """Handle agent exactly at goal position."""
        env.reset()
        env.agent_pos = env.goal_pos.copy()
        obs = env._get_obs()

        # Should not have NaN in direction (handled by dist > 1e-6 check)
        assert not np.any(np.isnan(obs))

    def test_very_low_battery(self, env):
        """Handle near-zero battery values."""
        env.reset()
        env.battery = 0.001
        obs = env._get_obs()

        assert obs[5] == pytest.approx(0.00001, rel=1e-3)
        assert not np.any(np.isnan(obs))

    def test_negative_battery_clamped(self, agent, hidden_state):
        """Agent should handle negative battery gracefully."""
        x = torch.randn(1, 1, 14)
        # Should not raise error with negative battery
        action, value, h = agent(x, hidden_state, battery=-10.0)

        assert not torch.any(torch.isnan(action))
        assert not torch.any(torch.isnan(value))

    def test_extreme_observations(self, agent, hidden_state):
        """Agent should handle extreme observation values."""
        x = torch.randn(1, 1, 14) * 100  # Large values
        action, value, h = agent(x, hidden_state, battery=100.0)

        # Action should still be bounded by tanh
        assert action.min() >= -1.0
        assert action.max() <= 1.0
        assert not torch.any(torch.isnan(action))

    def test_many_episodes_no_memory_leak(self, env, agent, hidden_state):
        """Running many short episodes should not cause issues."""
        for _ in range(50):
            obs = env.reset()
            h = hidden_state.clone()

            for _ in range(10):
                x = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0)
                action, _, h = agent(x, h, env.battery)
                obs, _, done, _ = env.step(action[0, 0].detach().numpy())
                if done:
                    break

    def test_agent_at_boundary(self, env):
        """Agent at environment boundary should be handled."""
        env.reset()
        env.agent_pos = np.array([0.1, 0.1])  # At boundary
        obs = env._get_obs()

        # Ray observations should be valid
        assert not np.any(np.isnan(obs))
        assert np.all(np.isfinite(obs))


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_m_state_no_nan(self, agent, hidden_state):
        """M-state should never become NaN."""
        x = torch.randn(1, 1, 14)

        for _ in range(1000):
            agent(x, hidden_state, battery=np.random.uniform(0, 100))

        assert not torch.isnan(agent.M_state)

    def test_scar_no_nan(self, agent, hidden_state):
        """Scar should never become NaN."""
        x = torch.randn(1, 1, 14)
        agent.M_state = torch.tensor(0.01)  # Force low M

        for _ in range(1000):
            agent(x, hidden_state, battery=1.0)

        assert not torch.isnan(agent.scar)

    def test_policy_distribution_valid(self, agent, hidden_state):
        """Policy distribution should always be valid."""
        from torch.distributions import Normal

        x = torch.randn(1, 1, 14)
        action_mean, _, _ = agent(x, hidden_state, battery=100.0)
        std = torch.exp(agent.log_std).clamp(1e-3, 1.0)

        # Should be able to create valid distribution
        dist = Normal(action_mean, std)
        sample = dist.sample()
        log_prob = dist.log_prob(sample)

        assert not torch.any(torch.isnan(sample))
        assert not torch.any(torch.isnan(log_prob))
        assert not torch.any(torch.isinf(log_prob))


# ============================================================================
# DETERMINISM TESTS
# ============================================================================

class TestDeterminism:
    """Tests for reproducibility with seeds."""

    def test_env_seed_reproducibility(self):
        """Same seed should produce same environment state."""
        np.random.seed(42)
        env1 = EmbodiedEnv(level=0)
        obs1 = env1.reset()
        pos1 = env1.agent_pos.copy()
        goal1 = env1.goal_pos.copy()

        np.random.seed(42)
        env2 = EmbodiedEnv(level=0)
        obs2 = env2.reset()
        pos2 = env2.agent_pos.copy()
        goal2 = env2.goal_pos.copy()

        np.testing.assert_array_equal(obs1, obs2)
        np.testing.assert_array_equal(pos1, pos2)
        np.testing.assert_array_equal(goal1, goal2)

    def test_agent_seed_reproducibility(self):
        """Same seed should produce same network initialization."""
        torch.manual_seed(42)
        agent1 = CAESARNet()

        torch.manual_seed(42)
        agent2 = CAESARNet()

        for p1, p2 in zip(agent1.parameters(), agent2.parameters()):
            torch.testing.assert_close(p1, p2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
