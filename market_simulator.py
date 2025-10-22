import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import config

class MarketSimulator(gym.Env):
    """
    A high-fidelity, Gymnasium-compliant market simulator for training the RL Governor.
    It replays historical data and incorporates domain randomization for robust training.
    """
    def __init__(self, historical_data: pd.DataFrame, initial_balance=1000):
        super().__init__()
        
        if historical_data.empty:
            raise ValueError("Historical data cannot be empty.")
            
        self.df = historical_data
        self.initial_balance = initial_balance
        
        # --- Domain Randomization Parameters ---
        self.base_fee = config.FEE_PERCENT
        self.base_slippage = config.SLIPPAGE_PERCENT
        self.latency_range_ms = (50, 500) # 50ms to 500ms latency

        # --- Gym Spaces ---
        # Observation space: market features + portfolio state
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=(len(self.df.columns) + 3,), # + balance, position, drawdown
                                            dtype=np.float32)
        # Action space: position size (0.0 to 1.0 of portfolio)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def _get_obs(self):
        """Constructs the observation from the current market and portfolio state."""
        market_features = self.df.iloc[self.current_step].values
        portfolio_state = np.array([self.balance, self.position_size, self.drawdown])
        return np.concatenate([market_features, portfolio_state]).astype(np.float32)

    def reset(self, seed=None, options=None):
        """Resets the environment for a new episode."""
        super().reset(seed=seed)
        
        # --- Domain Randomization ---
        # Randomize costs for this episode to make the agent more robust
        self.current_fee = np.random.uniform(self.base_fee * 0.8, self.base_fee * 1.2)
        self.current_slippage = np.random.uniform(self.base_slippage * 0.8, self.base_slippage * 1.2)
        
        # --- State Initialization ---
        self.current_step = 0
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.drawdown = 0
        self.position = 'neutral' # 'neutral', 'long', 'short'
        self.position_size = 0 # Fraction of portfolio
        self.entry_price = 0
        
        return self._get_obs(), {}

    def step(self, action):
        """Executes one time step within the environment."""
        target_size = action[0]
        
        # --- Stochastic Latency Simulation ---
        # Simulate a delay and get the price at that future point.
        # For simplicity in this discrete environment, we'll execute at the *next* step's open.
        self.current_step += 1
        if self.current_step >= len(self.df) -1:
            return self._get_obs(), 0, True, False, {} # End of data

        execution_price = self.df['open'].iloc[self.current_step]
        
        # --- PnL and Reward Calculation ---
        reward = 0
        
        # If in a position, calculate unrealized PnL as part of the reward
        if self.position != 'neutral':
            current_pnl = (execution_price / self.entry_price - 1) if self.position == 'long' else (self.entry_price / execution_price - 1)
            reward += current_pnl * self.position_size # Mark-to-market reward

        # Execute trade based on the delta between current size and target size
        if target_size != self.position_size:
            # For simplicity, we'll model a full exit and re-entry
            # Close current position
            if self.position_size > 0:
                exit_price = execution_price * (1 - self.current_slippage)
                pnl = (exit_price / self.entry_price - 1) if self.position == 'long' else (self.entry_price / exit_price - 1)
                self.balance += self.balance * self.position_size * pnl
                self.balance -= self.balance * self.position_size * self.current_fee # Exit fee
            
            # Open new position
            if target_size > 0:
                # This part needs the external signal (long/short) which is not part of the action space.
                # Assuming the RL agent only controls size, and an external signal controls direction.
                # This is a common setup for RL risk managers.
                # We'll need to pass the signal to the step function.
                # For now, we'll assume a random signal for demonstration.
                signal = np.random.choice(['long', 'short'])
                self.position = signal
                self.position_size = target_size
                self.entry_price = execution_price * (1 + self.current_slippage)
                self.balance -= self.balance * self.position_size * self.current_fee # Entry fee

        # Update portfolio metrics
        self.peak_balance = max(self.peak_balance, self.balance)
        self.drawdown = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0
        
        done = self.balance < self.initial_balance * 0.5 # Stop if 50% drawdown
        
        return self._get_obs(), reward, done, False, {}

class SimpleMarketSimulator:
    """A simple market simulator for training the RL execution agent."""
    def __init__(self, historical_data: pd.DataFrame, initial_balance=1000):
        if historical_data.empty:
            raise ValueError("Historical data cannot be empty.")
            
        self.df = historical_data
        self.initial_balance = initial_balance
        self.current_step = 0

    def reset(self):
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step >= len(self.df) -1:
            return None, True # End of data
        return self.df.iloc[self.current_step], False

    def get_observation(self):
        """Returns the current market observation."""
        return self.df.iloc[self.current_step]

    def execute_trade(self, action) -> dict:
        """Simulates the execution of a trade and returns the slippage."""
        # This is a simplified implementation. A real implementation would be more sophisticated.
        # For now, we will assume a fixed slippage.
        slippage = 0.001 # 0.1% slippage
        return {"slippage": slippage}

    def is_done(self) -> bool:
        """Checks if the simulation is done."""
        return self.current_step >= len(self.df) - 1