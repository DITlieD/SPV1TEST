"""
L2 Order Book Data Collector
=============================
Real-time L2 (full depth) order book collection from Bybit via WebSocket.

This is the data foundation for the Apex Causal Nexus (ACN):
- Micro-Causality Engine (MCE) - analyzes microstructure
- Chimera Engine - compares simulation to reality
- Topological Radar - detects regime shifts

Features:
- Full L2 depth reconstruction (50 levels)
- Incremental updates with snapshot recovery
- Multi-asset streaming
- Data persistence (Parquet format)
- Thread-safe access
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime
from collections import OrderedDict
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path
import threading
from queue import Queue
import time
import os
import csv
import numpy as np
try:
    import config as app_config
except ImportError:
    class ConfigFallback:
        DATA_DIR = 'data'
    app_config = ConfigFallback()

def load_historical_l2_data(symbol: str) -> pd.DataFrame:
    """
    Loads historical L2 snapshot data for training.
    ASSUMPTION: L2 data is stored in a pre-processed snapshot format.
    Format: timestamp, bid_p_0, bid_v_0, ask_p_0, ask_v_0, ...
    """
    try:
        base_symbol = symbol.split(':')[0].replace('/', '')
        # Adjust file extension based on actual storage format (CSV or Parquet)
        file_path = os.path.join(app_config.DATA_DIR, 'l2', f"{base_symbol}_L2_snapshots.csv")
        
        if not os.path.exists(file_path):
            logger.warning(f"L2 Data file not found: {file_path}")
            return pd.DataFrame()
            
        df_l2 = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
        
        # Verify essential columns exist
        required_cols = ['bid_p_0', 'bid_v_0', 'ask_p_0', 'ask_v_0']
        if not all(col in df_l2.columns for col in required_cols):
            logger.error(f"L2 data missing required L1 columns: {required_cols}")
            return pd.DataFrame()

        return df_l2
    except Exception as e:
        logger.error(f"Error loading L2 data for {symbol}: {e}")
        return pd.DataFrame()

logger = logging.getLogger(__name__)


class OrderBook:
    """
    Full L2 order book representation.

    Structure:
    - bids: {price: quantity} sorted descending
    - asks: {price: quantity} sorted ascending
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids: OrderedDict = OrderedDict()  # price -> qty (descending)
        self.asks: OrderedDict = OrderedDict()  # price -> qty (ascending)
        self.last_update_id = 0
        self.last_update_time = None
        self.initialized = False

    def apply_snapshot(self, bids: List[List], asks: List[List], update_id: int):
        """
        Apply full snapshot.

        Args:
            bids: List of [price, qty]
            asks: List of [price, qty]
            update_id: Sequence number
        """
        self.bids.clear()
        self.asks.clear()

        # Process bids (descending order)
        for price_str, qty_str in bids:
            price = float(price_str)
            qty = float(qty_str)
            if qty > 0:
                self.bids[price] = qty

        # Sort bids descending
        self.bids = OrderedDict(sorted(self.bids.items(), key=lambda x: -x[0]))

        # Process asks (ascending order)
        for price_str, qty_str in asks:
            price = float(price_str)
            qty = float(qty_str)
            if qty > 0:
                self.asks[price] = qty

        # Sort asks ascending
        self.asks = OrderedDict(sorted(self.asks.items(), key=lambda x: x[0]))

        self.last_update_id = update_id
        self.last_update_time = datetime.now()
        self.initialized = True

        logger.info(f"[L2 {self.symbol}] Snapshot applied: {len(self.bids)} bids, {len(self.asks)} asks")

    def apply_delta(self, bids: List[List], asks: List[List], update_id: int):
        """
        Apply incremental update.

        Args:
            bids: List of [price, qty] changes
            asks: List of [price, qty] changes
            update_id: Sequence number
        """
        if not self.initialized:
            logger.warning(f"[L2 {self.symbol}] Received delta before snapshot. Skipping.")
            return

        # Apply bid updates
        for price_str, qty_str in bids:
            price = float(price_str)
            qty = float(qty_str)

            if qty == 0:
                # Remove level
                self.bids.pop(price, None)
            else:
                # Update level
                self.bids[price] = qty

        # Re-sort bids
        self.bids = OrderedDict(sorted(self.bids.items(), key=lambda x: -x[0]))

        # Apply ask updates
        for price_str, qty_str in asks:
            price = float(price_str)
            qty = float(qty_str)

            if qty == 0:
                # Remove level
                self.asks.pop(price, None)
            else:
                # Update level
                self.asks[price] = qty

        # Re-sort asks
        self.asks = OrderedDict(sorted(self.asks.items(), key=lambda x: x[0]))

        self.last_update_id = update_id
        self.last_update_time = datetime.now()

    def get_best_bid(self) -> Optional[float]:
        """Get best bid price."""
        if self.bids:
            return next(iter(self.bids.keys()))
        return None

    def get_best_ask(self) -> Optional[float]:
        """Get best ask price."""
        if self.asks:
            return next(iter(self.asks.keys()))
        return None

    def get_mid_price(self) -> Optional[float]:
        """Get mid price."""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid and ask:
            return (bid + ask) / 2.0
        return None

    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid and ask:
            return ask - bid
        return None

    def get_snapshot(self, depth: int = 50) -> Dict:
        """
        Get current book state.

        Args:
            depth: Number of levels to include

        Returns:
            Dict with bids, asks, timestamp, mid_price
        """
        bids_list = list(self.bids.items())[:depth]
        asks_list = list(self.asks.items())[:depth]

        return {
            'symbol': self.symbol,
            'timestamp': self.last_update_time.isoformat() if self.last_update_time else None,
            'bids': [[p, q] for p, q in bids_list],
            'asks': [[p, q] for p, q in asks_list],
            'mid_price': self.get_mid_price(),
            'spread': self.get_spread(),
            'update_id': self.last_update_id
        }


class L2Collector:
    """
    Multi-asset L2 order book collector with persistence.
    """

    def __init__(self, assets: List[str], data_dir: str = "data", snapshot_interval: int = 60, cerebellum_link=None):
        """
        Args:
            assets: List of symbols (e.g., ['BTCUSDT', 'ETHUSDT'])
            data_dir: Directory for L2 data persistence
            snapshot_interval: Seconds between snapshot saves
            cerebellum_link: An instance of CerebellumLink to send data to.
        """
        self.assets = assets
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_interval = snapshot_interval
        self.cerebellum_link = cerebellum_link

        # Order books
        self.books: Dict[str, OrderBook] = {}
        for symbol in assets:
            self.books[symbol] = OrderBook(symbol)

        # WebSocket state
        self.ws_url = "wss://stream.bybit.com/v5/public/spot"
        self.running = False
        self.ws_task = None

        # Snapshot persistence
        self.snapshot_queue = Queue()
        self.persistence_thread = None

        logger.info(f"[L2Collector] Initialized for {len(assets)} assets")

    async def _connect_and_subscribe(self):
        """Establish WebSocket connection and subscribe to L2 streams."""
        retry_delay = 5

        while self.running:
            try:
                logger.info("[L2Collector] Connecting to Bybit WebSocket...")

                async with websockets.connect(self.ws_url) as ws:
                    logger.info("[L2Collector] Connected!")

                    # Subscribe to all assets (orderbook.50 = 50 levels)
                    subscription = {
                        "op": "subscribe",
                        "args": [f"orderbook.50.{symbol}" for symbol in self.assets]
                    }

                    await ws.send(json.dumps(subscription))
                    logger.info(f"[L2Collector] Subscribed to {len(self.assets)} orderbooks")

                    # Message processing loop
                    async for message in ws:
                        if not self.running:
                            break

                        try:
                            data = json.loads(message)
                            await self._process_message(data)
                        except Exception as e:
                            logger.error(f"[L2Collector] Message processing error: {e}")

            except Exception as e:
                logger.error(f"[L2Collector] WebSocket error: {e}. Reconnecting in {retry_delay}s...")
                await asyncio.sleep(retry_delay)

    async def _process_message(self, data: dict):
        """Process incoming WebSocket message."""
        # Check if it's orderbook data
        if data.get('topic', '').startswith('orderbook.50.'):
            symbol = data['topic'].split('.')[-1]

            if symbol not in self.books:
                return

            msg_type = data.get('type')
            book_data = data.get('data', {})

            if msg_type == 'snapshot':
                # Full snapshot
                bids = book_data.get('b', [])
                asks = book_data.get('a', [])
                update_id = book_data.get('u', 0)

                self.books[symbol].apply_snapshot(bids, asks, update_id)

            elif msg_type == 'delta':
                # Incremental update
                bids = book_data.get('b', [])
                asks = book_data.get('a', [])
                update_id = book_data.get('u', 0)

                self.books[symbol].apply_delta(bids, asks, update_id)

            # If a cerebellum link is provided, send the updated book state
            if self.cerebellum_link and self.books[symbol].initialized:
                self.cerebellum_link.send_l2_snapshot(self.books[symbol].get_snapshot())

    def _persistence_worker(self):
        """Background thread that saves snapshots to a CSV file for historical training."""
        logger.info("[L2Collector] Persistence thread started (CSV Mode)")
        
        headers = ['timestamp']
        for i in range(50): # 50 levels of depth
            headers.extend([f'bid_p_{i}', f'bid_v_{i}', f'ask_p_{i}', f'ask_v_{i}'])

        while self.running:
            try:
                time.sleep(10) # Save every 10 seconds

                for symbol in self.assets:
                    book = self.books.get(symbol)
                    if book and book.initialized:
                        snapshot = book.get_snapshot(depth=50)
                        
                        row = {'timestamp': snapshot['timestamp']}
                        bids = snapshot.get('bids', [])
                        asks = snapshot.get('asks', [])

                        for i in range(50):
                            if i < len(bids):
                                row[f'bid_p_{i}'] = bids[i][0]
                                row[f'bid_v_{i}'] = bids[i][1]
                            else:
                                row[f'bid_p_{i}'] = np.nan
                                row[f'bid_v_{i}'] = np.nan
                            
                            if i < len(asks):
                                row[f'ask_p_{i}'] = asks[i][0]
                                row[f'ask_v_{i}'] = asks[i][1]
                            else:
                                row[f'ask_p_{i}'] = np.nan
                                row[f'ask_v_{i}'] = np.nan

                        file_path = self.data_dir / f"{symbol}_L2_snapshots.csv"
                        file_exists = os.path.exists(file_path)
                        
                        with open(file_path, 'a', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=headers)
                            if not file_exists:
                                writer.writeheader()
                            writer.writerow(row)
                
                logger.info(f"Saved L2 snapshots for {len(self.assets)} assets.")

            except Exception as e:
                logger.error(f"[L2Collector] Persistence error: {e}")

    def start(self):
        """Start the L2 collector."""
        if self.running:
            logger.warning("[L2Collector] Already running")
            return

        logger.info("[L2Collector] Starting...")
        self.running = True

        # Start persistence thread
        self.persistence_thread = threading.Thread(target=self._persistence_worker, daemon=True)
        self.persistence_thread.start()

        # Start WebSocket in asyncio event loop
        def run_ws():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._connect_and_subscribe())

        self.ws_task = threading.Thread(target=run_ws, daemon=True)
        self.ws_task.start()

        logger.info("[L2Collector] Started successfully")

    def stop(self):
        """Stop the L2 collector."""
        logger.info("[L2Collector] Stopping...")
        self.running = False

        if self.ws_task:
            self.ws_task.join(timeout=5)

        if self.persistence_thread:
            self.persistence_thread.join(timeout=5)

        logger.info("[L2Collector] Stopped")

    def get_book(self, symbol: str) -> Optional[OrderBook]:
        """
        Get order book for a symbol.

        Args:
            symbol: Asset symbol

        Returns:
            OrderBook instance or None
        """
        return self.books.get(symbol)

    def get_all_snapshots(self, depth: int = 50) -> Dict[str, Dict]:
        """
        Get snapshots for all assets.

        Args:
            depth: Number of levels

        Returns:
            Dict of {symbol: snapshot}
        """
        snapshots = {}
        for symbol, book in self.books.items():
            if book.initialized:
                snapshots[symbol] = book.get_snapshot(depth)
        return snapshots

    def export_to_dataframe(self, symbol: str, depth: int = 10) -> Optional[pd.DataFrame]:
        """
        Export current order book to DataFrame (for analysis).

        Args:
            symbol: Asset symbol
            depth: Number of levels

        Returns:
            DataFrame with bid/ask levels
        """
        book = self.books.get(symbol)
        if not book or not book.initialized:
            return None

        snapshot = book.get_snapshot(depth)

        # Create bid DataFrame
        bids_df = pd.DataFrame(snapshot['bids'], columns=['price', 'qty'])
        bids_df['side'] = 'bid'

        # Create ask DataFrame
        asks_df = pd.DataFrame(snapshot['asks'], columns=['price', 'qty'])
        asks_df['side'] = 'ask'

        # Combine
        df = pd.concat([bids_df, asks_df], ignore_index=True)
        df['symbol'] = symbol
        df['timestamp'] = snapshot['timestamp']

        return df


# Standalone test/demo
if __name__ == "__main__":
    import sys
    # Add the project root to the Python path to allow importing 'config'
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Use all assets from the config for collection
    from config import ASSET_UNIVERSE, DATA_DIR
    assets_to_collect = [s.replace('/', '') for s in ASSET_UNIVERSE]

    collector = L2Collector(
        assets=assets_to_collect,
        data_dir=DATA_DIR,
        snapshot_interval=10 # Save every 10 seconds for historical data
    )

    collector.start()

    try:
        logger.info(f"[COLLECTOR] Now collecting L2 data for {len(assets_to_collect)} assets...")
        logger.info("Press CTRL+C to stop the collection process.")
        # Keep the main thread alive indefinitely
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("[COLLECTOR] Interrupt received, shutting down.")
    finally:
        collector.stop()
        logger.info("[COLLECTOR] Collection complete.")
