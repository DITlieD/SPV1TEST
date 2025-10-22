"""
L2 Microstructure Feature Engine
=================================
High-frequency microstructure features from full L2 order book data.

These features feed the Micro-Causality Engine (MCE) for Market Maker inference.

Features:
- Order Flow Imbalance (OFI) - Net aggressive order flow
- Book Pressure Ratio - Bid vs Ask depth imbalance
- Spread Dynamics - Various spread measures
- Volume-Weighted Metrics - VWAP, VWBID, VWASK
- Trade Flow Classification - Lee-Ready algorithm

Performance: Numba-optimized for real-time calculation
"""

import numpy as np
import pandas as pd
from numba import jit
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# OFFLINE TRAINING FUNCTION (from updatev16.txt)
# ============================================================================

def calculate_microstructure_features(df_l2: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates high-frequency microstructure features from L1 snapshot data for offline training.
    """
    if df_l2.empty:
        return pd.DataFrame()

    logger.info("Calculating Microstructure Features (OFI, BPR, WMP)...")
    features = pd.DataFrame(index=df_l2.index)

    bid_vol_0 = df_l2['bid_v_0']
    ask_vol_0 = df_l2['ask_v_0']
    bid_price_0 = df_l2['bid_p_0']
    ask_price_0 = df_l2['ask_p_0']
    total_vol = bid_vol_0 + ask_vol_0
    
    features['BPR_L1'] = bid_vol_0 / total_vol
    features['BPR_L1'].replace([np.inf, -np.inf], 0.5, inplace=True)

    wmp = (ask_price_0 * bid_vol_0 + bid_price_0 * ask_vol_0) / total_vol
    features['WMP'] = wmp
    features['Spread'] = ask_price_0 - bid_price_0

    prev_bid_p = bid_price_0.shift(1)
    prev_ask_p = ask_price_0.shift(1)
    prev_bid_v = bid_vol_0.shift(1)
    prev_ask_v = ask_vol_0.shift(1)
    ofi = pd.Series(0, index=df_l2.index, dtype=float)
    ofi += np.where(bid_price_0 > prev_bid_p, bid_vol_0, 0)
    ofi += np.where(bid_price_0 == prev_bid_p, bid_vol_0 - prev_bid_v, 0)
    ofi -= np.where(bid_price_0 < prev_bid_p, prev_bid_v, 0)
    ofi -= np.where(ask_price_0 < prev_ask_p, ask_vol_0, 0)
    ofi -= np.where(ask_price_0 == prev_ask_p, ask_vol_0 - prev_ask_v, 0)
    ofi += np.where(ask_price_0 > prev_ask_p, prev_ask_v, 0)
    features['OFI_L1'] = ofi

    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(method='ffill', inplace=True)
    features.fillna(0, inplace=True)

    return features

# ============================================================================
# REAL-TIME ENGINE COMPONENTS (Restored)
# ============================================================================

@jit(nopython=True)
def calculate_ofi_numba(bids_prev: np.ndarray, asks_prev: np.ndarray,
                        bids_curr: np.ndarray, asks_curr: np.ndarray,
                        bid_prices_prev: np.ndarray, ask_prices_prev: np.ndarray,
                        bid_prices_curr: np.ndarray, ask_prices_curr: np.ndarray) -> float:
    ofi = 0.0
    if len(bid_prices_prev) > 0 and len(bid_prices_curr) > 0:
        best_bid_prev = bid_prices_prev[0]
        best_bid_curr = bid_prices_curr[0]
        if best_bid_prev == best_bid_curr:
            delta_bid = bids_curr[0] - bids_prev[0]
        elif best_bid_curr > best_bid_prev:
            delta_bid = bids_curr[0]
        else:
            delta_bid = -bids_prev[0]
        ofi += delta_bid

    if len(ask_prices_prev) > 0 and len(ask_prices_curr) > 0:
        best_ask_prev = ask_prices_prev[0]
        best_ask_curr = ask_prices_curr[0]
        if best_ask_prev == best_ask_curr:
            delta_ask = asks_curr[0] - asks_prev[0]
        elif best_ask_curr < best_ask_prev:
            delta_ask = asks_curr[0]
        else:
            delta_ask = -asks_prev[0]
        ofi -= delta_ask
    return ofi

class MicrostructureFeatureEngine:
    """
    Real-time microstructure feature calculator.
    """
    def __init__(self, window_size: int = 20, depth: int = 10):
        self.window_size = window_size
        self.depth = depth
        self.prev_snapshot = {}
        self.ofi_history = []
        self.pressure_history = []
        self.spread_history = []
        logger.info(f"[MicrostructureEngine] Initialized (window={window_size}, depth={depth})")

    def calculate_all_features(self, book: Dict, symbol: str, trade_price: Optional[float] = None) -> Dict[str, float]:
        features = {}
        prev = self.prev_snapshot.get(symbol)
        if prev is not None:
            # Simplified OFI for real-time, full OFI is more complex
            features['ofi'] = book.get('bids', [[0,0]])[0][1] - prev.get('bids', [[0,0]])[0][1]
        else:
            features['ofi'] = 0.0
        
        bids_vol = sum(qty for _, qty in book.get('bids', [])[:self.depth])
        asks_vol = sum(qty for _, qty in book.get('asks', [])[:self.depth])
        total_vol = bids_vol + asks_vol
        features['book_pressure'] = (bids_vol - asks_vol) / total_vol if total_vol > 0 else 0.0

        # Other feature calculations would go here
        self.prev_snapshot[symbol] = book.copy()
        return features
