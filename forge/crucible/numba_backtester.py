# forge/crucible/numba_backtester.py
import pandas as pd
import numpy as np
import numba
import logging

logger = logging.getLogger(__name__)

@numba.jit(nopython=True)
def _run_backtest_core(open_prices, close_prices, entries, exits, 
                       initial_capital, profit_target_abs, 
                       fee_rate, spread_factor, slippage_factor):
    
    n_samples = len(open_prices)
    capital = initial_capital
    position = 0.0
    trade_count = 0
    max_drawdown = 0.0
    peak_capital = initial_capital
    time_to_target = n_samples

    for i in range(1, n_samples):
        # PnL Calculation (Mark-to-Market using Close prices)
        if position != 0.0:
            price_change = close_prices[i] - close_prices[i-1]
            capital += position * price_change
        
        # Update Drawdown
        if capital > peak_capital: peak_capital = capital
        drawdown = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0.0
        if drawdown > max_drawdown: max_drawdown = drawdown

        # Check Velocity Target (TTT)
        if capital >= profit_target_abs:
            time_to_target = i
            break # Optimization: Stop simulation early

        # --- Trade Execution Logic (at Open of the current bar) ---
        base_price = open_prices[i]
        
        if entries[i] and position == 0.0:
            # BUY Execution (Cross Spread + Slippage)
            execution_price = base_price * (1 + spread_factor + slippage_factor)
            # Sizing: All-in (Velocity focus)
            position = capital / execution_price * (1 - fee_rate)
            trade_count += 1

        elif exits[i] and position > 0.0:
            # SELL Execution (Cross Spread + Slippage)
            execution_price = base_price * (1 - spread_factor - slippage_factor)
            capital = position * execution_price * (1 - fee_rate)
            position = 0.0
            
    return capital, trade_count, max_drawdown, time_to_target

class VectorizedBacktester:
    def __init__(self, config: dict):
        self.config = config
        self.initial_capital = self.config.get('bt_initial_capital', 1000.0)
        self.target_pct = self.config.get('bt_velocity_target_pct', 2.0) / 100.0
        self.profit_target_abs = self.initial_capital * (1 + self.target_pct)
        
        # Costs
        self.fee_rate = self.config.get('bt_fees_pct', 0.055) / 100.0
        self.spread_factor = self.config.get('bt_spread_bps', 1.5) / 10000.0
        self.slippage_factor = self.config.get('bt_slippage_factor', 0.003)

    def run(self, df_context: pd.DataFrame, signals: pd.DataFrame) -> dict:
        
        # Data alignment and extraction
        signals_bool = signals[['entries', 'exits']].astype(bool)
        aligned_data = pd.concat([df_context[['open', 'close']], signals_bool], axis=1).dropna()
        
        if aligned_data.empty:
            return self.get_default_metrics()

        # Run Numba Core
        final_capital, trade_count, max_dd, ttt = _run_backtest_core(
            aligned_data['open'].values, aligned_data['close'].values,
            aligned_data['entries'].values, aligned_data['exits'].values,
            self.initial_capital, self.profit_target_abs,
            self.fee_rate, self.spread_factor, self.slippage_factor
        )

        # --- Fitness Calculation (The Velocity Objective) ---
        n_samples = len(aligned_data)
        
        if final_capital >= self.profit_target_abs:
             # Target achieved: High fitness based on speed (Range: [1.0, 2.0])
            ttt_score = 1.0 + ((n_samples - ttt) / n_samples)
        else:
            # Target not achieved: Fitness based on final PnL (Can be negative, < target_pct)
            final_pnl_pct = (final_capital - self.initial_capital) / self.initial_capital
            ttt_score = final_pnl_pct
        
        # Risk Penalties (Drawdown)
        risk_penalty = max_dd 
        velocity_fitness = ttt_score - risk_penalty

        return {
            "velocity_fitness": velocity_fitness,
            "final_capital": final_capital,
            "max_drawdown": max_dd,
        }

    def get_default_metrics(self):
        return {"velocity_fitness": -1e9, "final_capital": self.initial_capital, "max_drawdown": 0.0}