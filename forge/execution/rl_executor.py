
# forge/execution/rl_executor.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ExecutionEnv(gym.Env):
    """Custom Environment for training an RL agent for trade execution."""

    def __init__(self, market_simulator):
        super().__init__()

        self.market_simulator = market_simulator

        # Action space: 0: Market Buy, 1: Market Sell, 2: Hold
        self.action_space = spaces.Discrete(3)

        # Observation space
        self.observation_space = spaces.Dict({
            "ofi": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "book_pressure": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "liquidity_density": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "volatility": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "confidence": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })

    def _get_obs(self):
        # Get the latest market data from the simulator
        obs = self.market_simulator.get_observation()
        return {
            "ofi": np.array([obs['ofi']], dtype=np.float32),
            "book_pressure": np.array([obs['book_pressure']], dtype=np.float32),
            "liquidity_density": np.array([obs['liquidity_density']], dtype=np.float32),
            "volatility": np.array([obs['volatility']], dtype=np.float32),
            "confidence": np.array([obs['confidence']], dtype=np.float32),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.market_simulator.reset()
        return self._get_obs(), {}

    def step(self, action):
        # Execute the action in the market simulator
        execution_result = self.market_simulator.execute_trade(action)

        # Get the next observation
        observation = self._get_obs()

        # Calculate the reward
        # Reward is based on minimizing slippage
        reward = -execution_result['slippage']

        # Check if the episode is done
        terminated = self.market_simulator.is_done()

        return observation, reward, terminated, False, {}
