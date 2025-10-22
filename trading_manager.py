# trading_manager.py (V5.4 - Final Decoupled)
import os
import ccxt.pro as ccxt
import asyncio
import pandas as pd
from datetime import datetime

from forge.core.agent import V3Agent
from data_processing_v2 import get_aligned_mtf_data, add_all_features
from data_fetcher import get_market_data
from forge.data_processing.sentiment_engine import SentimentEngine
from forge.data_processing.data_validator import DataValidator
from forge.strategy.strategic_bias import StrategicBiasModel

class MultiAssetTrader:

    def __init__(self, asset_universe, hedge_manager, stop_event: asyncio.Event, app_config):
        print("[Trader] Initializing MultiAssetTrader...")
        self.asset_universe = asset_universe
        self.config = app_config
        self.exchange = self._setup_exchange()
        self.stop_event = stop_event
        self.hedge_manager = hedge_manager

        self.raw_data_queue = asyncio.Queue()
        self.features_queue = asyncio.Queue()
        self.sentiment_engine = SentimentEngine()
        self.validator = DataValidator()
        self.strategic_bias_model = StrategicBiasModel()
        self.workers = {}
        self.active_traders = set() # Keep track of assets with live trading loops
        
        print("\n--- Multi-Asset Trader Initialized (MTF Architecture) ---")

    async def _initialize_workers(self):
        """Asynchronously initializes V3Agent workers for each asset."""
        print("[Trader] Asynchronously creating V3Agent workers...")
        for symbol in self.asset_universe:
            try:
                print(f"[Trader] --> Fetching initial data for {symbol}...")
                initial_data = await asyncio.to_thread(self._get_initial_data, symbol)
                
                if initial_data.empty:
                    print(f"[Trader] WARNING: Could not get initial data for {symbol}. Agent will not be created.")
                    continue

                # --- Create a dedicated exchange instance for each agent ---
                print(f"[Trader] --> Configuring exchange for {symbol}...")
                symbol_key = symbol.replace('/','')
                agent_exchange_config = self.config.BYBIT_SUBACCOUNT_KEYS.get(symbol_key)
                if not agent_exchange_config or not agent_exchange_config.get('apiKey'):
                    print(f"[Trader] CRITICAL: Missing API key for {symbol} in config. Skipping agent creation.")
                    continue

                exchange_class = getattr(ccxt, self.config.ACTIVE_EXCHANGE)
                agent_exchange = exchange_class(agent_exchange_config)
                agent_exchange.set_sandbox_mode(self.config.BYBIT_USE_TESTNET)

                print(f"[Trader] --> Creating agent for {symbol}...")
                agent = await asyncio.to_thread(
                    V3Agent,
                    symbol, 
                    agent_exchange, # Pass the dedicated exchange instance
                    self.config
                )
                self.workers[symbol] = agent
                print(f"[Trader] --> Agent for {symbol} created successfully.")
            except Exception as e:
                print(f"[Trader] CRITICAL: Failed to create agent for {symbol}: {e}")

    def _get_initial_data(self, symbol):
        try:
            # MTFA: Use microstructure (1m) for live execution
            df = get_market_data(symbol, self.config.TIMEFRAMES['microstructure'], limit=2000)
            df = add_all_features(df)
            df['label'] = 0
            return df
        except Exception as e:
            print(f"Error getting initial data for {symbol}: {e}")
            return pd.DataFrame()

    def _setup_exchange(self):
        """Initializes a basic CCXT exchange instance for public data."""
        try:
            exchange_class = getattr(ccxt, self.config.ACTIVE_EXCHANGE)
            exchange = exchange_class()
            if self.config.ACTIVE_EXCHANGE == 'bybit':
                exchange.set_sandbox_mode(self.config.BYBIT_USE_TESTNET)
            print(f"[Trader] Basic exchange '{self.config.ACTIVE_EXCHANGE}' setup complete.")
            return exchange
        except Exception as e:
            print(f"[Trader] CRITICAL: Failed to setup basic exchange: {e}")
            raise

    async def _watch_ohlcv_for_symbol(self, symbol, timeframe):
        """Continuously watches for new OHLCV candles for a single symbol."""
        print(f"[{symbol}] Starting OHLCV watcher for {timeframe}...")
        while not self.stop_event.is_set():
            try:
                candles = await self.exchange.watch_ohlcv(symbol, timeframe)
                if candles:
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    df.attrs['symbol'] = symbol
                    print(f"[{symbol}] Received new {timeframe} candle at {df.index[-1]}.")
                    await self.raw_data_queue.put(df)
            except Exception as e:
                print(f"[{symbol}] Error in OHLCV watcher: {e}. Retrying in 10s...")
                await asyncio.sleep(10)

    async def _data_ingestion_loop(self):
        """Creates and manages a watcher task for each *active* asset."""
        print(f"[Trader] Starting data ingestion loop for active traders: {list(self.active_traders)}")
        # MTFA: Watch microstructure (1m) for live execution
        tasks = [self._watch_ohlcv_for_symbol(symbol, self.config.TIMEFRAMES['microstructure']) for symbol in self.active_traders]
        if tasks:
            await asyncio.gather(*tasks)
        else:
            print("[Trader] Data ingestion loop has no active traders to monitor. Waiting for activation signal.")
            # If there are no tasks, we wait indefinitely until the loop is cancelled and restarted.
            await self.stop_event.wait()
        print("[Trader] Data ingestion loop stopped.")

    async def _signal_and_execution_loop(self):
        """Processes features and triggers agent decisions."""
        print("[Trader] Starting signal and execution loop...")
        while not self.stop_event.is_set():
            try:
                features_df = await self.features_queue.get()
                symbol = features_df.attrs.get('symbol')
                if symbol and symbol in self.workers:
                    print(f"[{symbol}] Processing new features...")
                    sentiment_score, _ = self.sentiment_engine.get_sentiment()
                    strategic_bias = self.strategic_bias_model.get_bias(features_df)
                    drift_signal = 0.0 
                    agent = self.workers[symbol]
                    await agent.decide_and_execute(features_df, sentiment_score, drift_signal, strategic_bias)
                self.features_queue.task_done()
            except Exception as e:
                print(f"[Trader] Error in signal/execution loop: {e}")

    async def run(self):
        """The main entry point for the MultiAssetTrader service."""
        await self._initialize_workers()
        print("[Trader] Starting main trading loops...")
        try:
            self.ingestion_task = asyncio.create_task(self._data_ingestion_loop())
            feature_task = asyncio.create_task(self._feature_engineering_loop())
            execution_task = asyncio.create_task(self._signal_and_execution_loop())
            await asyncio.gather(self.ingestion_task, feature_task, execution_task)
        except Exception as e:
            print(f"[Trader] A critical error occurred in the main run loop: {e}")
        finally:
            print("[Trader] Main trading loops stopped.")
            if hasattr(self, 'exchange') and self.exchange:
                await self.exchange.close()

    async def _feature_engineering_loop(self):
        """Processes raw candles into feature dataframes."""
        print("[Trader] Starting feature engineering loop...")
        while not self.stop_event.is_set():
            try:
                candle_df = await self.raw_data_queue.get()
                symbol = candle_df.attrs.get('symbol')
                
                if symbol:
                    print(f"[{symbol}] Engineering features for new candle...")
                    # get_aligned_mtf_data is async, so we can await it directly
                    df_aligned = await get_aligned_mtf_data(symbol, self.config)
                    
                    if not df_aligned.empty and self.validator.validate(df_aligned, symbol):
                        await self.features_queue.put(df_aligned)
                
                self.raw_data_queue.task_done()
            except Exception as e:
                print(f"[Trader] Error in feature engineering loop: {e}")

    async def rescan_and_update_worker(self, symbol: str):
        """Reloads models and activates trading for a specific worker agent."""
        if symbol in self.workers:
            print(f"[{symbol}] Received rescan signal. Reloading models for agent...")
            agent = self.workers[symbol]
            await agent._reload_models()
            print(f"[{symbol}] Agent models reloaded successfully.")
            
            if symbol not in self.active_traders:
                print(f"[{symbol}] ACTIVATING trading loops.")
                self.active_traders.add(symbol)
                # Restart the data ingestion loop to include the new symbol
                if hasattr(self, 'ingestion_task') and not self.ingestion_task.done():
                    self.ingestion_task.cancel()
                self.ingestion_task = asyncio.create_task(self._data_ingestion_loop())
        else:
            print(f"[{symbol}] WARNING: Received rescan signal but no agent found for this symbol.")

    def get_all_worker_statuses(self):
        """Gets the status of all V3Agent workers."""
        active_bots = []
        for symbol, worker in self.workers.items():
            active_bots.append(worker.get_status())
        
        return {
            "asset_universe": self.asset_universe,
            "active_bots": active_bots
        }
