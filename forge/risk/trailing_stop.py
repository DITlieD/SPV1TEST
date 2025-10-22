# forge/risk/trailing_stop.py

class ATRTrailingStop:
    """
    A self-contained class for calculating an ATR-based trailing stop loss.
    Initializes the peak/trough from the entry price to ensure correct trailing from the start.
    """
    def __init__(self, entry_price: float, initial_stop: float, atr_multiplier: float, direction: str = 'long'):
        self.current_stop = initial_stop
        self.atr_multiplier = atr_multiplier
        self.direction = direction
        
        if direction == 'long':
            self.peak_price = entry_price 
        else:
            self.trough_price = entry_price

    def update(self, high_price: float, low_price: float, atr_val: float) -> float:
        """
        Updates the trailing stop based on the current bar's data.
        """
        if self.direction == 'long':
            self.peak_price = max(self.peak_price, high_price)
            new_stop = self.peak_price - (atr_val * self.atr_multiplier)
            self.current_stop = max(self.current_stop, new_stop)
        else: # short
            self.trough_price = min(self.trough_price, low_price)
            new_stop = self.trough_price + (atr_val * self.atr_multiplier)
            self.current_stop = min(self.current_stop, new_stop)
        return self.current_stop
