# forge/data_processing/microstructure.py
import pandas as pd

def calculate_order_book_features(order_book: dict, depth=10) -> dict:
    """
    Calculates Order Book Imbalance (OBI) and Bid-Ask Spread.

    Args:
        order_book (dict): A level 2 order book snapshot from CCXT.
        depth (int): The number of levels to consider for the imbalance calculation.

    Returns:
        dict: A dictionary containing 'obi' and 'spread'.
    """
    if not order_book or not order_book['bids'] or not order_book['asks']:
        return {'obi': 0.5, 'spread': 0.0} # Return neutral values

    # Get top N levels
    bids = order_book['bids'][:depth]
    asks = order_book['asks'][:depth]

    # Spread
    best_bid = bids[0][0]
    best_ask = asks[0][0]
    spread = (best_ask - best_bid) / ((best_ask + best_bid) / 2)

    # Order Book Imbalance
    bid_volume = sum(vol for price, vol in bids)
    ask_volume = sum(vol for price, vol in asks)
    total_volume = bid_volume + ask_volume
    
    obi = bid_volume / total_volume if total_volume > 0 else 0.5

    return {'obi': obi, 'spread': spread}
