# forge/execution/rl_governor_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GovernorEnv(gym.Env):
    """
    Custom Environment for training the RL Governor.
    The Governor's role is to adjust the risk of a proposed trade.
    """
    def __init__(self, market_simulator):
        super().__init__()
        self.market_simulator = market_simulator

        # Action space: A continuous value to multiply the trade size by.
        # e.g., 0.5 = half risk, 1.0 = normal risk, 1.5 = 1.5x risk
        self.action_space = spaces.Box(low=0.5, high=1.5, shape=(1,), dtype=np.float32)

        # Observation space: The information the Governor uses to make a decision.
        self.observation_space = spaces.Dict({
            "volatility": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "confidence": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "kelly_fraction": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "sharpe_ratio": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        })

    def _get_obs(self):
        obs = self.market_simulator.get_observation()
        return {
            "volatility": np.array([obs.get('volatility', 0)], dtype=np.float32),
            "confidence": np.array([obs.get('confidence', 0.5)], dtype=np.float32),
            "kelly_fraction": np.array([obs.get('kelly_fraction', 0)], dtype=np.float32),
            "sharpe_ratio": np.array([obs.get('sharpe_ratio', 0)], dtype=np.float32),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.market_simulator.reset()
        return self._get_obs(), {}

    def step(self, action):
        # The action is the risk multiplier from the RL agent.
        risk_multiplier = action[0]

        # The market simulator executes a trade based on its internal logic,
        # but adjusted by the governor's risk multiplier.
        trade_result = self.market_simulator.execute_trade(risk_multiplier)

        # Get the next observation.
        observation = self._get_obs()

        # --- REWARD CALCULATION ---
        # The reward is the PnL of the trade, scaled by the risk taken.
        pnl = trade_result.get('pnl', 0)
        
        # Scale reward by the action.
        # If pnl is positive, higher risk (action > 1) is rewarded more.
        # If pnl is negative, higher risk is penalized more.
        reward = pnl * (1 + (risk_multiplier - 1.0) * np.sign(pnl))

        terminated = self.market_simulator.is_done()
        
        return observation, reward, terminated, False, {}
