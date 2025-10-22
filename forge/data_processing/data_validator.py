# forge/data_processing/data_validator.py
import pandas as pd
import numpy as np

class DataValidator:
    """
    A class to perform data quality checks on OHLCV dataframes.
    """
    def __init__(self, max_spike_pct=0.20, max_nan_pct=0.10):
        """
        Args:
            max_spike_pct (float): The maximum allowed single-candle spike (e.g., 0.20 for 20%).
            max_nan_pct (float): The maximum percentage of NaN values allowed in a column.
        """
        self.max_spike_pct = max_spike_pct
        self.max_nan_pct = max_nan_pct

    def validate(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        Runs a series of validation checks on the dataframe.

        Returns:
            bool: True if the data is valid, False otherwise.
        """
        print(f"({symbol}) [Validator] Running data quality checks...")
        
        # 1. Check for NaNs or infinite values
        if df.isnull().values.any() or np.isinf(df.values).any():
            nan_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
            if nan_pct > self.max_nan_pct:
                print(f"({symbol}) [Validator] ❌ FAILED: Excessive NaN/inf values found ({nan_pct:.2%}).")
                return False
            else:
                print(f"({symbol}) [Validator] ✅ PASSED: NaN/inf check (found {nan_pct:.2%}, below threshold).")

        # 2. Check for impossible prices (zero or negative)
        if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
            print(f"({symbol}) [Validator] ❌ FAILED: Zero or negative price values found.")
            return False
        print(f"({symbol}) [Validator] ✅ PASSED: Price value check.")

        # 3. Check for stale data (no price change over a long period)
        if (df['close'].diff(periods=10).fillna(0) == 0).all():
            print(f"({symbol}) [Validator] ❌ FAILED: Data appears stale (no price change).")
            return False
        print(f"({symbol}) [Validator] ✅ PASSED: Stale data check.")

        # 4. Check for extreme price spikes
        price_change = df['close'].pct_change().abs()
        if (price_change > self.max_spike_pct).any():
            print(f"({symbol}) [Validator] ❌ FAILED: Extreme price spike detected (>{self.max_spike_pct:.2%}).")
            return False
        print(f"({symbol}) [Validator] ✅ PASSED: Price spike check.")

        print(f"({symbol}) [Validator] --- All data quality checks passed ---")
        return True
