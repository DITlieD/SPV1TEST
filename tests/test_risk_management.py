import pytest
from datetime import date, timedelta
from risk_management import CircuitBreaker
import config

@pytest.fixture
def circuit_breaker():
    """Returns a new CircuitBreaker instance for each test."""
    return CircuitBreaker()

def test_initial_state(circuit_breaker):
    """Test that the breaker is initially open for trades."""
    assert circuit_breaker.is_trade_allowed(current_equity=10000, peak_equity=10000) == True
    assert circuit_breaker.is_daily_breaker_active == False
    assert circuit_breaker.is_peak_breaker_active == False

def test_daily_drawdown_trigger(circuit_breaker):
    """Test that the daily drawdown limit trips the breaker."""
    initial_equity = 10000
    config.DAILY_LOSS_LIMIT = 0.05
    
    # First check, sets the start_of_day_equity
    assert circuit_breaker.is_trade_allowed(initial_equity, initial_equity) == True
    
    # Simulate a loss that exceeds the limit
    breach_equity = initial_equity * (1 - config.DAILY_LOSS_LIMIT - 0.01)
    assert circuit_breaker.is_trade_allowed(breach_equity, initial_equity) == False
    assert circuit_breaker.is_daily_breaker_active == True

def test_daily_drawdown_reset(circuit_breaker):
    """Test that the daily breaker resets on a new day."""
    initial_equity = 10000
    config.DAILY_LOSS_LIMIT = 0.05
    
    # Trip the breaker
    circuit_breaker.is_trade_allowed(initial_equity, initial_equity)
    breach_equity = initial_equity * 0.94
    circuit_breaker.is_trade_allowed(breach_equity, initial_equity)
    assert circuit_breaker.is_daily_breaker_active == True

    # Simulate a new day
    circuit_breaker.last_reset_date = date.today() - timedelta(days=1)
    
    # Check if it resets
    assert circuit_breaker.is_trade_allowed(breach_equity, initial_equity) == True
    assert circuit_breaker.is_daily_breaker_active == False
    assert circuit_breaker.start_of_day_equity == breach_equity

def test_peak_drawdown_trigger(circuit_breaker):
    """Test that the peak drawdown halt trips the breaker."""
    peak_equity = 20000
    config.PEAK_DRAWDOWN_HALT = 0.30
    
    # Simulate a loss that exceeds the limit
    breach_equity = peak_equity * (1 - config.PEAK_DRAWDOWN_HALT - 0.01)
    assert circuit_breaker.is_trade_allowed(breach_equity, peak_equity) == False
    assert circuit_breaker.is_peak_breaker_active == True

def test_peak_drawdown_does_not_reset(circuit_breaker):
    """Test that the peak drawdown halt is permanent and does not reset daily."""
    peak_equity = 20000
    config.PEAK_DRAWDOWN_HALT = 0.30
    
    # Trip the breaker
    breach_equity = peak_equity * 0.69
    circuit_breaker.is_trade_allowed(breach_equity, peak_equity)
    assert circuit_breaker.is_peak_breaker_active == True

    # Simulate a new day
    circuit_breaker.last_reset_date = date.today() - timedelta(days=1)
    
    # Breaker should remain active
    assert circuit_breaker.is_trade_allowed(breach_equity, peak_equity) == False
    assert circuit_breaker.is_peak_breaker_active == True
