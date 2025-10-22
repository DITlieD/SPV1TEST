import vectorbt as vbt
import numpy as np
import pandas as pd
from scipy.stats import norm

def calculate_psr(returns: pd.Series, sharpe_ratio: float, min_trades=20) -> float:
    """Calculates the Probabilistic Sharpe Ratio (PSR)."""
    if returns.empty or len(returns) < min_trades:
        return 0.0
    
    if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
        return 0.0

    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    n = len(returns)

    psr_numerator = (sharpe_ratio * np.sqrt(n - 1))
    psr_denominator = np.sqrt(1 - skewness * sharpe_ratio + ((kurtosis - 1) / 4) * sharpe_ratio**2)

    if psr_denominator == 0 or np.isnan(psr_denominator):
        return 0.0

    probabilistic_sharpe_ratio = norm.cdf(psr_numerator / psr_denominator)
    return probabilistic_sharpe_ratio if not np.isnan(probabilistic_sharpe_ratio) else 0.0

class VectorizedBacktester:
    """
    A high-fidelity backtesting engine using vectorbt to simulate performance
    with realistic market frictions.

    ULTRA-AGGRESSIVE MFT (1-Minute Timeframe):
    - Realistic costs: 0.055% taker fees, 2 bps spread, 0.005 slippage
    - VELOCITY: 10% profit target, 0.5% stop-loss (20:1 R/R), 10 min timeout
    - Philosophy: Trade fast, fail fast, reap losers, evolve winners
    - Reaping: Models below $0 equity get immediate failure (-1e9 fitness)
    """
    def __init__(self,
                 commission_pct=0.00055,      # MFT: Bybit taker fee (0.055%)
                 slippage_factor=0.005,       # MFT: Realistic slippage for 1m
                 spread_bps=2.0,              # MFT: Typical spread (2 basis points = 0.02%)
                 max_participation_rate=0.1,
                 initial_capital=200,         # Ultra-aggressive: $200 per bot
                 profit_target_pct=10.0,      # ULTRA MFT: 10% profit target - grow FAST!
                 stop_loss_pct=0.5,           # Tight risk: 0.5% stop loss = 20:1 R/R
                 max_bars_in_trade=10):       # VELOCITY: Max 10 bars (10 min on 1m) - rapid fire!
        self.commission_pct = commission_pct
        self.slippage_factor = slippage_factor
        self.spread_factor = spread_bps / 10000.0  # Convert bps to decimal
        self.max_participation_rate = max_participation_rate
        self.initial_capital = initial_capital
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_bars_in_trade = max_bars_in_trade

    def run(self, data: pd.DataFrame, signals: pd.DataFrame) -> dict:
        # ... (data prep is the same) ...
        if data.empty or signals.empty:
            return self.get_default_metrics()

        atr_col = next((c for c in data.columns if c.startswith('ATRr')), None)
        if not atr_col or atr_col not in data.columns:
            raise ValueError("ATR column (e.g., 'ATRr_14') not found in data for slippage calculation.")

        dynamic_cost_amount = data[atr_col] * self.slippage_factor
        dynamic_slippage_pct = dynamic_cost_amount / data['open']
        dynamic_slippage_pct.fillna(0, inplace=True)

        if 'volume' not in data.columns:
            raise ValueError("'volume' column not found in data for liquidity constraint calculation.")

        max_order_size = data['volume'] * self.max_participation_rate

        # MFT: Combine fees (commission + spread)
        total_fees = self.commission_pct + self.spread_factor

        try:
            freq = pd.infer_freq(data.index)
            if freq is None:
                freq = '1T'  # MFT: Default to 1-minute

            # DIAGNOSTIC: Check signals
            import random
            if random.random() < 0.001:
                print(f"[BACKTESTER] entries.sum()={signals['entries'].sum()}, exits.sum()={signals['exits'].sum()}, data_len={len(data)}")

            # MFT: Portfolio with stop-loss and take-profit
            # Note: VectorBT's sl_stop/tp_stop are percentage-based (e.g., 0.005 = 0.5%)
            # Timeout is handled by live trading system (adaptive 15-60 min based on volatility)
            portfolio = vbt.Portfolio.from_signals(
                close=data['close'],
                entries=signals['entries'],
                exits=signals['exits'],
                price=data['open'],
                slippage=dynamic_slippage_pct,
                fees=total_fees,  # MFT: Commission + spread
                max_size=max_order_size,
                freq=freq,
                init_cash=self.initial_capital,
                sl_stop=self.stop_loss_pct / 100.0,  # MFT: Stop-loss (e.g., 0.5% = 0.005)
                tp_stop=self.profit_target_pct / 100.0  # MFT: Take-profit per trade (3% = 0.03)
            )

            returns = portfolio.returns()
            if returns.empty:
                if random.random() < 0.001:
                    print(f"[BACKTESTER] WARNING: returns.empty! entries={signals['entries'].sum()}, exits={signals['exits'].sum()}")
                return self.get_default_metrics()

            log_returns = np.log(returns + 1)
            log_wealth = log_returns.sum()
            sharpe = portfolio.sharpe_ratio()

            # VELOCITY UPGRADE: Calculate time-to-target and ttt_fitness
            total_return = portfolio.total_return() * 100.0  # Convert to percentage
            equity_curve = portfolio.value()

            # MFT REAPING: Check if model ever goes below $0 (instant death)
            min_equity = equity_curve.min()
            if min_equity < 0:
                # REAPING: Model went bankrupt, immediate failure
                return {
                    "total_return": -1.0,
                    "sharpe_ratio": -10.0,
                    "probabilistic_sharpe_ratio": 0.0,
                    "max_drawdown": 1.0,
                    "total_trades": portfolio.trades.count(),
                    "win_rate": 0.0,
                    "log_wealth": -1e9,
                    "time_to_target": -1,
                    "ttt_fitness": -1e9,  # REAPED!
                    "returns": returns
                }

            # Calculate time to reach profit target (ULTRA MFT: 10% - grow FAST!)
            target_capital = self.initial_capital * (1.0 + self.profit_target_pct / 100.0)
            time_to_target = -1  # Default: target not reached

            for i in range(len(equity_curve)):
                if equity_curve.iloc[i] >= target_capital:
                    time_to_target = i
                    break

            # Calculate TTT fitness (matches NumbaTurboBacktester formula)
            base_penalty = float(len(data))

            if time_to_target > 0:
                # Reached target
                speed_component = -float(time_to_target)
                if total_return > 0:
                    profit_multiplier = 1.0 + (total_return / 100.0)
                    ttt_fitness = speed_component * profit_multiplier
                else:
                    loss_penalty = abs(total_return) * 100.0
                    ttt_fitness = speed_component - loss_penalty
            else:
                # Didn't reach target
                if total_return > 0:
                    ttt_fitness = -(base_penalty * 0.5) + (total_return * 10.0)
                else:
                    ttt_fitness = -base_penalty + (total_return * 10.0)

            return {
                "total_return": portfolio.total_return(),
                "sharpe_ratio": sharpe,
                "probabilistic_sharpe_ratio": calculate_psr(returns, sharpe),
                "max_drawdown": portfolio.max_drawdown(),
                "total_trades": portfolio.trades.count(),
                "win_rate": portfolio.trades.win_rate(),
                "log_wealth": log_wealth if np.isfinite(log_wealth) else -1e9,
                "time_to_target": int(time_to_target),
                "ttt_fitness": float(ttt_fitness),
                "returns": returns
            }
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"[Backtester] CRITICAL ERROR during vbt.Portfolio simulation: {e}", exc_info=True)
            return self.get_default_metrics()

    def get_default_metrics(self):
        return {
            "total_return": -1.0,
            "sharpe_ratio": -10.0,
            "probabilistic_sharpe_ratio": 0.0,
            "max_drawdown": 1.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "log_wealth": -1e9,
            "time_to_target": -1,
            "ttt_fitness": -1e9,
            "returns": pd.Series(dtype=np.float64)
        }
