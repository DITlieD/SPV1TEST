# forge/data_processing/labeling.py
import pandas as pd
import numpy as np

def get_triple_barrier_labels(df: pd.DataFrame, pt_sl: list = [2.0, 2.0], max_hold_bars: int = 20) -> pd.Series:
    """
    Implements a symmetrical Triple-Barrier Method for labeling financial time series.
    This version checks for both long and short opportunities from each candle.

    Args:
        df (pd.DataFrame): DataFrame with 'high', 'low', 'close', and 'ATRr_14' (ATR) columns.
        pt_sl (list): Symmetrical multipliers for profit-take and stop-loss barriers based on ATR. [pt, sl].
        max_hold_bars (int): Maximum number of bars to hold a position.

    Returns:
        pd.Series: A series of labels (1 for buy, 2 for sell, 0 for hold).
    """
    print(f"[Labeling] Creating labels with Symmetrical Triple-Barrier (PT/SL: {pt_sl[0]}*ATR, Max Hold: {max_hold_bars} bars)")
    
    close_prices = df['close']
    atr = df['ATRr_14'] if 'ATRr_14' in df.columns else df['close'] * 0.01
    
    labels = pd.Series(0, index=df.index)
    
    for i in range(len(df) - max_hold_bars):
        entry_price = close_prices.iloc[i]
        current_atr = atr.iloc[i]
        
        # Symmetrical barriers for both long and short
        upper_barrier = entry_price + (current_atr * pt_sl[0])
        lower_barrier = entry_price - (current_atr * pt_sl[1])
        
        for j in range(1, max_hold_bars + 1):
            future_high = df['high'].iloc[i + j]
            future_low = df['low'].iloc[i + j]
            
            # Check if upper barrier is hit first (long opportunity)
            if future_high >= upper_barrier:
                labels.iloc[i] = 1 # Buy signal
                break
            # Check if lower barrier is hit first (short opportunity)
            elif future_low <= lower_barrier:
                labels.iloc[i] = 2 # Sell signal
                break
        # If loop completes, vertical barrier was hit (label remains 0)

    buy_count = (labels == 1).sum()
    sell_count = (labels == 2).sum()
    hold_count = (labels == 0).sum()
    print(f"[Labeling] Distribution: BUY={buy_count}, SELL={sell_count}, HOLD={hold_count}")
    
    # Address extreme imbalance if one class is nearly absent
    total_signals = buy_count + sell_count
    if total_signals > 0 and (buy_count / total_signals < 0.1 or sell_count / total_signals < 0.1):
        print("[Labeling] WARNING: Extreme imbalance detected. Consider adjusting barrier parameters.")

    return labels

# Keep the old function but rename it to avoid breaking anything that might still use it.
def create_categorical_labels(df: pd.DataFrame, horizon: int = 10, threshold: float = 0.005) -> pd.Series:
    """
    DEPRECATED: Use get_triple_barrier_labels for more robust labeling.
    This function is kept for legacy compatibility.
    """
    return get_triple_barrier_labels(df)