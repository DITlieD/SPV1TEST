"""
Opponent Strategy Templates for Chimera Engine
"""
import numpy as np
import pandas_ta as ta

def simple_rsi_opponent(df, params):
    """Simple RSI strategy template."""
    rsi_period, buy_threshold, sell_threshold = params
    rsi = df.ta.rsi(length=int(rsi_period))

    signals = np.zeros(len(df))
    signals[rsi < buy_threshold] = 1
    signals[rsi > sell_threshold] = -1

    return signals

opponent_param_bounds = [(10, 30), (20, 40), (60, 80)]
