# forge/modeling/volatility_engine.py
import pandas as pd
import numpy as np
from arch import arch_model

import warnings
from arch.utility.exceptions import ConvergenceWarning

class VolatilityEngine:
    """
    Provides rolling volatility forecasts using an EGARCH(1,1) model.
    EGARCH is chosen for its ability to capture the leverage effect, where
    negative shocks have a greater impact on volatility than positive ones.
    """
    def __init__(self, window_size=500, forecast_horizon=1):
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.model = None
        print("[VolatilityEngine] Initialized.")

    def get_forecast(self, df_strategic: pd.DataFrame) -> float:
        """
        Fits the EGARCH model on a rolling window of the most recent data
        and returns the next-step volatility forecast.
        """
        close_col = 'strategic_close'
        if close_col not in df_strategic.columns:
            close_col = 'close' # Fallback to the standard column name
            if close_col not in df_strategic.columns:
                raise ValueError("Required column 'strategic_close' or 'close' not found.")

        if len(df_strategic) < self.window_size:
            return 0.05 # Return a default, moderate volatility if not enough data

        # Scale returns to percentage points for better optimizer convergence
        returns = df_strategic[close_col].pct_change().dropna() * 100
        returns.replace([np.inf, -np.inf], np.nan, inplace=True)
        returns.dropna(inplace=True)
        rolling_returns = returns.iloc[-self.window_size:]

        if rolling_returns.var() < 1e-6: # If there's virtually no variance, return a low default
            return 0.01

        try:
            # The 'rescale=False' is important as we've already scaled the data
            self.model = arch_model(rolling_returns, vol='EGARCH', p=1, o=1, q=1, dist='t', rescale=False)
            
            # Fit the model with increased iterations and warnings suppressed
            result = self.model.fit(disp='off', 
                                     show_warning=False, 
                                     options={'maxiter': 200})

            forecast = result.forecast(horizon=self.forecast_horizon, reindex=False)
            
            # The variance is in terms of (percent returns)^2. We need to scale it back.
            forecasted_std_dev = np.sqrt(forecast.variance.iloc[-1, 0]) / 100
            
            # Annualize the standard deviation
            annualized_vol = forecasted_std_dev * np.sqrt(252 * 6) # Assuming 4h data
            
            return annualized_vol if not np.isnan(annualized_vol) else 0.05
        except Exception as e:
            # print(f"[VolatilityEngine] WARNING: EGARCH model failed to fit: {e}. Returning default.")
            return 0.05
