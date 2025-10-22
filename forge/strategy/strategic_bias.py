# forge/strategy/strategic_bias.py
import pandas as pd

class StrategicBiasModel:
    """
    Determines the high-level strategic bias (macro trend) based on the
    strategic (high-timeframe) market data.
    """
    def __init__(self, fast_ema_period=50, slow_ema_period=200):
        self.fast_ema_period = fast_ema_period
        self.slow_ema_period = slow_ema_period
        print("[StrategicBias] Initialized with EMA periods:", fast_ema_period, slow_ema_period)

    def get_bias(self, df_strategic: pd.DataFrame) -> str:
        """
        Calculates the strategic bias from the latest strategic data.

        Args:
            df_strategic: A DataFrame containing the strategic (e.g., 4h) data,
                          with columns prefixed accordingly (e.g., 'strategic_close').

        Returns:
            str: 'LONG_ONLY', 'SHORT_ONLY', or 'NEUTRAL'.
        """
        close_col = 'strategic_close'
        if close_col not in df_strategic.columns:
            raise ValueError(f"Required column '{close_col}' not found in strategic DataFrame.")

        if len(df_strategic) < self.slow_ema_period:
            return 'NEUTRAL' # Not enough data to determine a bias

        # Calculate EMAs
        fast_ema = df_strategic[close_col].ewm(span=self.fast_ema_period, adjust=False).mean().iloc[-1]
        slow_ema = df_strategic[close_col].ewm(span=self.slow_ema_period, adjust=False).mean().iloc[-1]

        # Determine bias
        if fast_ema > slow_ema * 1.005: # Add a small buffer to prevent noise
            bias = 'LONG_ONLY'
        elif fast_ema < slow_ema * 0.995: # Add a small buffer
            bias = 'SHORT_ONLY'
        else:
            bias = 'NEUTRAL'
            
        print(f"[StrategicBias] Determined bias: {bias} (Fast EMA: {fast_ema:.2f}, Slow EMA: {slow_ema:.2f})")
        return bias
