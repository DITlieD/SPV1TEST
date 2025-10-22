"""
Circuit Breaker System for the Fallen God Trading Pipeline.

This file provides a stateless rules engine for the core safety circuit breakers.
It is designed to work with the real-time V3 architecture.
"""

from datetime import date
import config

class CircuitBreaker:
    """
    A stateless engine that checks for breaches of daily and peak drawdown limits.
    It receives the current state on every call, ensuring it always uses live data.
    """
    def __init__(self):
        self.is_daily_breaker_active = False
        self.is_peak_breaker_active = False
        self.start_of_day_equity = None
        self.last_reset_date = None

    def check_and_reset_daily_breaker(self, current_equity: float):
        """Checks if a new day has started and resets the daily tracker."""
        today = date.today()
        if self.last_reset_date != today:
            self.last_reset_date = today
            self.start_of_day_equity = current_equity
            if self.is_daily_breaker_active:
                print("🌅 NEW DAY: Daily circuit breaker has been reset.")
                self.is_daily_breaker_active = False

    def is_trade_allowed(self, current_equity: float, peak_equity: float) -> bool:
        """
        The main decision function. Checks all circuit breaker rules.

        Args:
            current_equity: The real-time equity from the exchange.
            peak_equity: The real-time peak equity from the exchange.

        Returns:
            False if any breaker is active, True otherwise.
        """
        self.check_and_reset_daily_breaker(current_equity)

        # Check for Daily Drawdown
        if self.start_of_day_equity:
            daily_drawdown = (self.start_of_day_equity - current_equity) / self.start_of_day_equity
            if not self.is_daily_breaker_active and daily_drawdown >= config.DAILY_LOSS_LIMIT:
                print(f"🚨 DAILY CIRCUIT BREAKER TRIPPED! Daily loss > {config.DAILY_LOSS_LIMIT:.0%}. Halting new trades.")
                self.is_daily_breaker_active = True

        # Check for Peak Equity Drawdown
        if peak_equity > 0:
            peak_drawdown = (peak_equity - current_equity) / peak_equity
            if not self.is_peak_breaker_active and peak_drawdown >= config.PEAK_DRAWDOWN_HALT:
                print(f"🔥🔥 PEAK DRAWDOWN HALT! Total drawdown > {config.PEAK_DRAWDOWN_HALT:.0%}. System requires human review.")
                self.is_peak_breaker_active = True
        
        # Return final decision
        if self.is_daily_breaker_active or self.is_peak_breaker_active:
            return False
        
        return True

class ActionShield:
    """
    Provides a final, pre-trade safety check based on microstructure volatility.
    This is the last line of defense against entering a trade in a chaotic, unpredictable moment.
    """
    def __init__(self):
        # These limits are loaded from the main config file
        self.velocity_cap = config.ACTION_SHIELD_LIMITS.get("velocity_cap", 0.05)

    def is_safe_to_trade(self, microstructure_volatility: float) -> bool:
        """
        Checks if the immediate (e.g., 1-minute) volatility is within acceptable limits.

        Args:
            microstructure_volatility (float): A measure of recent, low-timeframe volatility.

        Returns:
            bool: False if volatility exceeds the cap, True otherwise.
        """
        if microstructure_volatility > self.velocity_cap:
            return False
        return True
