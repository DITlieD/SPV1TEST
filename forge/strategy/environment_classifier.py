
import pandas as pd
import numpy as np
import pandas_ta

class EnvironmentClassifier:
    """
    Analyzes market data to classify the current environment based on volatility and trend.
    This classification allows the optimization process to adapt its goals.
    """
    def __init__(self, volatility_period=20, trend_period=14, volatility_q=0.75, trend_q=0.5):
        """
        Initializes the classifier.

        Args:
            volatility_period (int): Lookback period for calculating ATR (volatility).
            trend_period (int): Lookback period for calculating Choppiness Index (trend).
            volatility_q (float): The quantile to define "high" vs "low" volatility.
            trend_q (float): The quantile to define "trending" vs "ranging".
        """
        self.vol_period = volatility_period
        self.trend_period = trend_period
        self.vol_quantile = volatility_q
        self.trend_quantile = trend_q

    def _choppiness_index(self, df: pd.DataFrame) -> pd.Series:
        """Calculates the Choppiness Index."""
        atr = df.ta.atr(length=1, append=False)
        sum_atr = atr.rolling(window=self.trend_period).sum()
        
        highest_high = df['high'].rolling(window=self.trend_period).max()
        lowest_low = df['low'].rolling(window=self.trend_period).min()
        
        log_val = np.log10(sum_atr / (highest_high - lowest_low))
        log_period = np.log10(self.trend_period)
        
        return 100 * log_val / log_period

    def classify(self, df_market_data: pd.DataFrame) -> str:
        """
        Classifies the most recent market state.

        Args:
            df_market_data (pd.DataFrame): DataFrame with high, low, close columns.

        Returns:
            str: A string descriptor of the market environment (e.g., "High_Volatility_Trend").
        """
        if df_market_data.empty or len(df_market_data) < max(self.vol_period, self.trend_period):
            return "Undefined"

        # Calculate metrics
        volatility = df_market_data.ta.atr(length=self.vol_period, append=False)
        trendiness = self._choppiness_index(df_market_data)

        # Get the most recent values
        current_vol = volatility.iloc[-1]
        current_trend = trendiness.iloc[-1]

        # Determine thresholds from historical quantiles
        vol_threshold = volatility.quantile(self.vol_quantile)
        trend_threshold = trendiness.quantile(self.trend_quantile)

        # Classify based on thresholds
        vol_state = "High_Volatility" if current_vol > vol_threshold else "Low_Volatility"
        trend_state = "Trend" if current_trend < trend_threshold else "Reversion" # Lower chop = more trend

        environment = f"{vol_state}_{trend_state}"
        print(f"[EnvClassifier] Current Environment: {environment} (Vol: {current_vol:.2f}, Trend: {current_trend:.2f})")
        
        return environment
