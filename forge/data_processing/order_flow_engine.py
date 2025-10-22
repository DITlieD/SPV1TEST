
# forge/data_processing/order_flow_engine.py

import pandas as pd
import numpy as np

def calculate_order_flow_imbalance(order_book: dict) -> float:
    """Calculates the Order Flow Imbalance (OFI) from the L2 order book."""
    if not order_book or not order_book.get('bids') or not order_book.get('asks'):
        return 0.0

    best_bid_volume = order_book['bids'][0][1]
    best_ask_volume = order_book['asks'][0][1]

    denominator = best_bid_volume + best_ask_volume
    if denominator == 0:
        return 0.0

    return (best_bid_volume - best_ask_volume) / denominator

def calculate_book_pressure(order_book: dict, levels: int = 5) -> float:
    """Calculates the book pressure from the top N levels of the order book."""
    if not order_book or not order_book.get('bids') or not order_book.get('asks'):
        return 0.0

    bid_volumes = sum([level[1] for level in order_book['bids'][:levels]])
    ask_volumes = sum([level[1] for level in order_book['asks'][:levels]])

    denominator = bid_volumes + ask_volumes
    if denominator == 0:
        return 0.0

    return bid_volumes / denominator

def calculate_liquidity_density(order_book: dict, levels: int = 5) -> float:
    """Calculates the liquidity density from the top N levels of the order book."""
    if not order_book or not order_book.get('bids') or not order_book.get('asks'):
        return 0.0

    bid_volumes = sum([level[1] for level in order_book['bids'][:levels]])
    ask_volumes = sum([level[1] for level in order_book['asks'][:levels]])

    # Assuming a tick size of 0.01 for this example
    # In a real scenario, this should be fetched from the exchange
    tick_size = 0.01

    return (bid_volumes + ask_volumes) / (levels * tick_size)
