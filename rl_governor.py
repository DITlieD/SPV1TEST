import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import config
import torch
import torch.nn as nn
from torch.distributions import Beta
from torch.optim import Adam
from sklearn.preprocessing import RobustScaler

# --- Deep RL Implementation (PPO) ---

class ActorCriticNetwork(nn.Module):
    """A deep neural network for the Actor-Critic agent."""
    def __init__(self, n_inputs, n_outputs):
        super(ActorCriticNetwork, self).__init__()

        # Shared layers
        self.shared_net = nn.Sequential(
            nn.Linear(n_inputs, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

        # Actor head: outputs parameters for a Beta distribution (alpha, beta)
        self.actor_head = nn.Sequential(
            nn.Linear(256, n_outputs * 2), # 2 outputs for alpha and beta
            nn.Softplus() # Ensures alpha and beta are positive
        )

        # Critic head: outputs a single value for the state
        self.critic_head = nn.Linear(256, 1)

    def forward(self, x):
        shared_out = self.shared_net(x)
        
        # Actor
        action_params = self.actor_head(shared_out) + 1e-6 # Add epsilon for stability
        alpha, beta = action_params.chunk(2, dim=-1)
        dist = Beta(alpha, beta)
        
        # Critic
        value = self.critic_head(shared_out)
        
        return dist, value

class PPOAgent:
    """PPO Agent containing the training logic."""
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, n_epochs=10, batch_size=64, device='cpu'):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device

        self.network = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = Adam(self.network.parameters(), lr=lr)

    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            state = state.to(self.device)
            dist, value = self.network(state)
            if deterministic:
                action = dist.mean
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action).sum(axis=-1)
        return action.cpu(), log_prob.cpu(), value.cpu()

    def learn(self, states, actions, log_probs_old, rewards, dones, values):
        # Move all data to the device
        states = states.to(self.device)
        actions = actions.to(self.device)
        log_probs_old = log_probs_old.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        values = values.to(self.device)

        # Calculate advantages using Generalized Advantage Estimation (GAE)
        advantages = torch.zeros_like(rewards, device=self.device)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + values[:-1]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO Update
        for _ in range(self.n_epochs):
            for i in range(0, len(states), self.batch_size):
                batch_indices = range(i, min(i + self.batch_size, len(states)))
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_log_probs_old = log_probs_old[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                dist, critic_value = self.network(batch_states)
                log_probs_new = dist.log_prob(batch_actions).sum(axis=-1)
                entropy = dist.entropy().mean()

                # Actor (Policy) Loss
                ratio = (log_probs_new - batch_log_probs_old).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic (Value) Loss
                critic_loss = nn.MSELoss()(critic_value.squeeze(), batch_returns.squeeze())

                # Total Loss
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

# --- Environment and Governor ---


class RLGovernorEnv(gym.Env):
    def __init__(self, df_features: pd.DataFrame, p_meta_signals: pd.Series, hmm_regime_probs: np.ndarray,
                 ensemble_weights: np.ndarray, drift_signal: float):
        super(RLGovernorEnv, self).__init__()

        try:
            self.df = df_features.copy()
            # --- REAL, DEFINITIVE SANITIZATION ON ENTRY ---
            self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
            self.df.ffill(inplace=True)
            self.df.bfill(inplace=True)
            self.df.dropna(inplace=True)
        except Exception as e:
            print(f"--- FATAL ERROR IN RL_GOVERNOR_ENV: DATA PREPARATION ---")
            raise e

        try:
            self.p_meta_signals = p_meta_signals
            self.hmm_regime_probs = hmm_regime_probs
            self.ensemble_weights = ensemble_weights
            self.drift_signal = drift_signal
            self.initial_balance = 10000
            self.mdd_threshold = getattr(config, 'PEAK_DRAWDOWN_HALT', 0.5)
            self.commission = config.FEE_PERCENT
            self.slippage = config.SLIPPAGE_PERCENT
            self.feature_columns = [c for c in self.df.columns if c not in ['open','high','low','close','volume','label']]
        except Exception as e:
            print(f"--- FATAL ERROR IN RL_GOVERNOR_ENV: PARAMETER ASSIGNMENT ---")
            raise e

        try:
            # --- Robust Feature Scaling ---
            variances = self.df[self.feature_columns].var()
            near_zero_var_cols = variances[variances < 1e-6].index
            if not near_zero_var_cols.empty:
                self.feature_columns = [col for col in self.feature_columns if col not in near_zero_var_cols]
            self.scaler = RobustScaler()
            self.scaler.fit(self.df[self.feature_columns])
        except Exception as e:
            print(f"--- FATAL ERROR IN RL_GOVERNOR_ENV: SCALER FITTING ---")
            raise e

        try:
            # Define observation and action spaces
            n_features = len(self.feature_columns)
            self.observation_space = spaces.Box(low=-100, high=100, shape=(n_features + 4,), dtype=np.float32) # Added clipping to the space definition
            self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        except Exception as e:
            print(f"--- FATAL ERROR IN RL_GOVERNOR_ENV: SPACE DEFINITION ---")
            raise e
        
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.current_step = 0
        self.current_position = 0
        self.trade_entry_price = 0
        self.trade_size = 0
        
        obs = self._next_observation()
        obs = self._next_observation()
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0) # Sanitize first observation
        return obs, {}

    def _next_observation(self):
        # Get feature row as a DataFrame to preserve column names for the scaler
        features_df = self.df[self.feature_columns].iloc[[self.current_step]]
        
        # Scale features to prevent numerical instability
        scaled_features = self.scaler.transform(features_df).flatten()
        
        drawdown = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0.0
        state = np.concatenate([scaled_features, [self.balance / self.initial_balance, self.current_position, self.trade_size, drawdown]])
        
        # --- BRUTE-FORCE FINAL SANITIZATION ---
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        state = np.clip(state, -100, 100)
            
        return torch.tensor(state, dtype=torch.float32)

    def step(self, action):
        # --- DEFINITIVE, NUMERICALLY STABLE IMPLEMENTATION ---
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            return self._next_observation(), 0.0, True, False, {}

        # 1. Sanitize Action and State
        size_fraction = torch.clamp(action[0], 0.0, 1.0).item()
        current_price = self.df['close'].iloc[self.current_step]
        previous_wealth = self.balance

        if not np.isfinite(current_price) or current_price <= 0:
            return self._next_observation(), -1.0, True, False, {} # Terminate on bad data

        # 2. Determine Target Position
        m1_signal = self.p_meta_signals.iloc[self.current_step]
        target_position = 0
        if m1_signal > 0.55: target_position = 1
        elif m1_signal < 0.45: target_position = -1

        # 3. Execute Trades and Sanitize PnL
        if self.current_position != 0 and self.current_position != target_position:
            exit_price = current_price * (1 - self.slippage)
            if self.trade_entry_price > 1e-9 and exit_price > 1e-9:
                pnl_pct = (exit_price / self.trade_entry_price - 1) if self.current_position == 1 else (self.trade_entry_price / exit_price - 1)
            else:
                pnl_pct = 0.0
            
            pnl_pct = np.nan_to_num(pnl_pct, nan=0.0, posinf=0.0, neginf=0.0)
            self.balance += self.balance * self.trade_size * pnl_pct
            self.balance -= self.balance * self.trade_size * self.commission
            self.current_position = 0

        if self.current_position == 0 and target_position != 0:
            self.current_position = target_position
            self.trade_size = size_fraction
            self.trade_entry_price = current_price * (1 + self.slippage)
            self.balance -= self.balance * self.trade_size * self.commission

        # 4. Sanitize Balance and Calculate Numerically Stable Reward
        if not np.isfinite(self.balance) or self.balance <= 0:
            self.balance = 1e-9 # Reset to a tiny positive number to prevent log(0)

        reward = 0.0
        if previous_wealth > 1e-9 and self.balance > 1e-9:
            reward = np.log(self.balance / previous_wealth)

        reward = np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        reward = np.clip(reward, -1.0, 1.0) # Final safeguard

        # 5. Update State and Check for Termination
        self.peak_balance = max(self.peak_balance, self.balance)
        drawdown = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0.0
        terminated = self.balance < self.initial_balance * 0.5 or drawdown > self.mdd_threshold
        
        return self._next_observation(), reward, terminated, False, {}

class RLGovernor:
    def __init__(self, env: RLGovernorEnv, device='auto'):
        self.env = env
        
        # --- CORE FIX: Translate 'gpu' to 'cuda' for PyTorch ---
        resolved_device = 'cpu'
        if device == 'auto':
            resolved_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device == 'gpu':
            resolved_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            resolved_device = device # e.g., 'cpu'

        self.device = torch.device(resolved_device)
        
        print(f"[RL Governor] Initializing agent on device: {self.device}")

        if self.env:
            self.agent = PPOAgent(env.observation_space.shape[0], env.action_space.shape[0], device=self.device)
        else:
            self.agent = None

    def train(self, total_timesteps=10000):
        print(f"[RL Governor] Training Deep RL model for {total_timesteps} timesteps...")
        if not self.env or not self.agent:
            print("[RL Governor] ERROR: Environment not set. Cannot train.")
            return

        # Trajectory buffer
        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
        
        obs, _ = self.env.reset()
        for t in range(total_timesteps):
            action, log_prob, value = self.agent.get_action(obs)
            
            states.append(obs)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)

            obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            rewards.append(torch.tensor([reward], dtype=torch.float32))
            dones.append(torch.tensor([done], dtype=torch.float32))

            if done:
                obs, _ = self.env.reset()

        # Get the value of the last state
        _, _, last_value = self.agent.get_action(obs)
        values.append(last_value)

        # Convert lists to tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        log_probs = torch.stack(log_probs)
        rewards = torch.stack(rewards)
        dones = torch.stack(dones)
        values = torch.stack(values)

        self.agent.learn(states, actions, log_probs, rewards, dones, values)
        print("[RL Governor] Deep RL model training complete.")

    def predict(self, observation):
        if self.agent is None: return 0.0
        # Ensure observation is a tensor and on the correct device
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32)
        observation = observation.to(self.device)
        action, _, _ = self.agent.get_action(observation, deterministic=True)
        return action.item()

    def save(self, file_path):
        if self.agent: torch.save(self.agent.network.state_dict(), file_path)

    @classmethod
    def load(cls, file_path, env=None):
        instance = cls(env=env)
        if instance.agent:
            instance.agent.network.load_state_dict(torch.load(file_path))
        return instance