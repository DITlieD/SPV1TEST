# governor_market_simulator.py
import numpy as np
import pandas as pd

class GovernorMarketSimulator:
    """
    A simulator for training the RL Governor.
    It simulates trade outcomes based on historical data and a base strategy.
    """
    def __init__(self, df: pd.DataFrame, base_signals: pd.Series):
        self.df = df
        self.base_signals = base_signals
        self.current_step = 0
        self.max_steps = len(df) - 1
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.in_position = False
        self.entry_price = 0

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.in_position = False
        self.entry_price = 0
        return self.get_observation()

    def get_observation(self):
        # Return a dictionary of the current market state
        return self.df.iloc[self.current_step].to_dict()

    def execute_trade(self, risk_multiplier: float):
        """
        Simulates a trade based on the base strategy's signal and the governor's risk multiplier.
        """
        pnl = 0
        signal = self.base_signals.iloc[self.current_step]
        current_price = self.df['close'].iloc[self.current_step]

        if self.in_position:
            # If in position, check for exit signal
            if signal == 2: # Exit signal
                pnl = (current_price - self.entry_price) * (self.balance / self.entry_price)
                self.balance += pnl
                self.in_position = False
        else:
            # If not in position, check for entry signal
            if signal == 1: # Entry signal
                # The governor's risk multiplier is applied here
                trade_size = self.balance * risk_multiplier
                self.entry_price = current_price
                self.in_position = True
        
        self.current_step += 1
        
        return {'pnl': pnl}

    def is_done(self):
        return self.current_step >= self.max_steps
