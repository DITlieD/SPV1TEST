# forge/risk/position_sizer.py
import numpy as np

def calculate_fractional_kelly(
    win_prob: float, 
    win_loss_ratio: float, 
    fraction: float = 0.2
) -> float:
    """
    Calculates a fractional Kelly Criterion for position sizing.

    Args:
        win_prob (float): The model's calibrated probability of a winning trade (p).
        win_loss_ratio (float): The ratio of the average win to the average loss (b).
        fraction (float): The fraction of the Kelly bet to take (e.g., 0.2 for 20%).

    Returns:
        float: The suggested position size as a fraction of portfolio equity.
    """
    if win_loss_ratio <= 0:
        return 0.0 # Cannot size a position with no edge

    # Simplified Kelly formula: f* = p - q / b
    # where p = win_prob, q = 1 - win_prob, b = win_loss_ratio
    kelly_fraction = win_prob - ((1 - win_prob) / win_loss_ratio)

    # Apply the safety fraction and ensure the result is between 0 and 1
    sized_fraction = fraction * kelly_fraction
    
    return np.clip(sized_fraction, 0, 1)
