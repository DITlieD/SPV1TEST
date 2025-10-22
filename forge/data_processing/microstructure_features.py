# forge/data_processing/microstructure_features.py
import pandas as pd
import numpy as np

def get_trade_flow_imbalance(df_1m: pd.DataFrame, window=30) -> pd.Series:
    """
    Calculates the Trade Flow Imbalance (TFI) using the Tick Rule on 1-minute data.
    TFI measures the net buying vs. selling pressure.

    Args:
        df_1m (pd.DataFrame): DataFrame with 1-minute OHLCV data.
        window (int): The rolling window for the TFI EMA.

    Returns:
        pd.Series: A Series containing the TFI signal.
    """
    print("[TFI] Calculating Trade Flow Imbalance...")
    
    # 1. Classify ticks using the Tick Rule
    price_change = df_1m['close'].diff()
    ticks = np.ones(len(df_1m)) # Start by assuming all are buys
    ticks[price_change < 0] = -1 # Mark as sells if price decreased
    # If price is unchanged, use the previous tick's classification (forward fill)
    ticks[price_change == 0] = 0
    ticks = pd.Series(ticks, index=df_1m.index).replace(0, method='ffill')

    # 2. Calculate signed volume (Trade Flow)
    trade_flow = df_1m['volume'] * ticks

    # 3. Calculate Trade Flow Imbalance (TFI)
    # TFI = EMA(Signed Volume) / EMA(Total Volume)
    ema_signed_volume = trade_flow.ewm(span=window, adjust=False).mean()
    ema_total_volume = df_1m['volume'].ewm(span=window, adjust=False).mean()

    tfi = ema_signed_volume / (ema_total_volume + 1e-9) # Add epsilon to avoid division by zero
    
    print("[TFI] TFI calculation complete.")
    return tfi.rename('tfi')
