import asyncio
import ccxt.pro as ccxt
import pandas as pd
import numpy as np
import json
import os
import time
import logging
import config
from concurrent.futures import ProcessPoolExecutor
import multiprocessing # Ensure multiprocessing is imported
import joblib # FIX: Import joblib
from typing import List
from data_fetcher import get_market_data
from data_processing_v2 import get_aligned_mtf_data, run_graph_feature_pipeline, process_and_align_mtfa
from forge.crucible.arena_manager import ArenaManager, CrucibleAgent
from forge.crucible.watchtower import Watchtower
from forge.crucible.pit_crew import PitCrew  # VELOCITY UPGRADE: Hot-swap architecture
from forge.core.agent import V3Agent
from forge.data_processing.feature_factory import FeatureFactory
from forge.strategy.dynamic_regimes import DynamicRegimeModel
from singularity_engine import SingularityEngine
from forge.utils.pipeline_status import PipelineStatus
from forge.monitoring.immune_system import ImmuneSystem
from forge.analysis.complexity_manager import KolmogorovComplexityManager
from forge_worker import run_forge_process
from stable_baselines3 import PPO
from cerebellum_link import CerebellumLink
from forge.data_processing.l2_collector import L2Collector
from forge.modeling.micro_causality import (
    MCEFeatureEngine, MCEFeatureNormalizer, MCE_TFT_Model, MCE_RealtimePipeline
)
import torch
from forge.data_processing.microstructure_features_l2 import MicrostructureFeatureEngine
from forge.modeling.macro_causality import InfluenceMapperPipeline
from forge.acp.acp_engines import TopologicalRadar

def _initialize_regime_worker(symbol: str, exchange_name: str, sandbox_mode: bool) -> tuple:
    # (Ensure single-threading enforcement OMP_NUM_THREADS=1 is present)
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    # (Ensure necessary imports are inside the function)
    import asyncio
    import ccxt.pro as ccxt
    import pandas as pd
    import numpy as np
    from data_fetcher import get_market_data
    from forge.data_processing.feature_factory import FeatureFactory
    from forge.strategy.predictive_regimes import PredictiveRegimeModel
    import logging
    import config

    print(f"[{symbol}] Starting initial regime model training (Worker Forced Single Thread)...")

    exchange = None
    loop = None
    try:
        # Initialize exchange using the passed parameters
        try:
            exchange_class = getattr(ccxt, exchange_name)
            exchange = exchange_class({'options': {'defaultType': 'spot'}, 'enableRateLimit': True})
            if sandbox_mode:
                exchange.set_sandbox_mode(config.BYBIT_USE_TESTNET)
        except Exception as e:
            print(f"[{symbol}] CRITICAL: Failed to initialize exchange {exchange_name}: {e}")
            return symbol, None

        # Setup asyncio loop for the worker
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Fetch data
        sanitized_symbol = symbol.split(':')[0].replace('/', '')
        data_path = os.path.join('data', f"{sanitized_symbol}_1h_raw.csv")

        if os.path.exists(data_path):
            df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
        else:
            df = loop.run_until_complete(get_market_data(symbol, '1h', 1000, exchange_instance=exchange))

        if df is None or df.empty:
            print(f"[{symbol}] WARNING: Could not fetch data for regime model initialization.")
            return symbol, None
        
        # FeatureFactory processing
        worker_logger = logging.getLogger(f"RegimeWorker_{os.getpid()}")
        ff = FeatureFactory({symbol: df.copy()}, logger=worker_logger)

        # VELOCITY FIX: Feature Synthesizer runs here (once per asset per cycle)
        # This creates evolved_0..evolved_4 features needed by regime models
        # Still skipped during regime init and MTFA, so only runs once per asset

        processed_data = ff.create_all_features(app_config=config, exchange=None, skip_feature_evolution=False)
        df_features = processed_data[symbol]

        # PredictiveRegimeModel fitting
        regime_model = PredictiveRegimeModel(n_components=4)
        
        # CRITICAL FIX: Exclude any potential data leakage columns before training
        features_for_regime = df_features.select_dtypes(include=np.number).drop(columns=['open', 'high', 'low', 'close', 'volume', 'label', 'target_return'], errors='ignore')
        
        if features_for_regime.empty:
             print(f"[{symbol}] WARNING: No suitable features for regime model.")
             return symbol, None

        regime_model.fit(features_for_regime)
        print(f"[{symbol}] Regime model fitted successfully.")
        
        # Save the model
        sanitized_symbol = symbol.split(':')[0].replace('/', '')
        filepath = os.path.join(config.MODEL_DIR, f"{sanitized_symbol}_HMM.joblib")
        regime_model.save(filepath)
        print(f"[{symbol}] Saved HMM model to {filepath}")

        return symbol, regime_model

    except Exception as e:
        print(f"[{symbol}] ERROR: Failed to initialize regime model: {e}")
        import traceback
        traceback.print_exc()
        return symbol, None
    finally:
        if exchange and loop:
            try:
                loop.run_until_complete(exchange.close())
            except Exception:
                pass

import dill

# Define Priority Levels
PRIORITY_HIGH = 1   # Watchtower (Reaping)
PRIORITY_MEDIUM = 5 # Seeding
PRIORITY_LOW = 10   # Proactive

import multiprocessing as mp

class CrucibleEngine:
    def __init__(self, app_config, is_main_instance: bool = False):
        self.config = app_config
        self.is_main_instance = is_main_instance
        self.stop_event = asyncio.Event()
        
        # --- Shared State for Multiprocessing ---
        self.manager = mp.Manager()
        self.shared_state = self.manager.dict()
        self.shared_state['inferred_opponent_params'] = None
        
        self.logger = self._setup_logger("CrucibleEngine")
        self.singularity_logger = self._setup_logger("SingularityEngine")

        # --- Initialize Core Components First ---
        self.forge_queue = asyncio.PriorityQueue()
        self.active_forge_tasks = set()
        self.task_counter = 0  # Unique counter for queue tiebreaking
        self.exchange = self._setup_exchange()
        self.watchtower = Watchtower(drawdown_limit=0.5)  # MFT: Reap at $0 (100% DD), not 50% DD
        self.pit_crew = PitCrew(app_config, logger=self.logger)
        self.pipeline_status = PipelineStatus()
        self.immune_system = None
        self.generation_counters = {symbol: 0 for symbol in self.config.ASSET_UNIVERSE}
        self.raw_data_queue = asyncio.Queue()
        self.executor = None
        self.kcm = KolmogorovComplexityManager()

        self.l2_collector = L2Collector(
            assets=[s.replace('/', '') for s in self.config.ASSET_UNIVERSE],
            snapshot_interval=5
        )
        self.logger.info("L2 Data Collector initialized.")

        self.mce_feature_engine = MCEFeatureEngine()
        
        normalizer_path = os.path.join(config.MODEL_DIR, "mce_normalizer.pkl")
        if os.path.exists(normalizer_path):
            try:
                self.mce_normalizer = joblib.load(normalizer_path)
                self.logger.info("Pre-fitted MCEFeatureNormalizer loaded successfully.")
            except Exception as e:
                self.logger.error(f"Failed to load MCEFeatureNormalizer: {e}. Using a new instance.")
                self.mce_normalizer = MCEFeatureNormalizer()
        else:
            self.logger.warning("MCEFeatureNormalizer not found. Using a new instance. Please run train_mce_normalizer.py.")
            self.mce_normalizer = MCEFeatureNormalizer()
        
        self.mce_tft_model = MCE_TFT_Model()
        tft_model_path = "models/mce_tft_model.pth"
        
        if os.path.exists(tft_model_path):
            try:
                # This is a simplification. In a real scenario, you would need to
                # re-create the model with the same parameters as when it was trained,
                # or save/load the entire MCE_TFT_Model object.
                # For now, we assume the path contains a loadable model object.
                self.mce_tft_model.model = torch.load(tft_model_path)
                self.mce_tft_model.model.to(config.DEVICE)
                self.mce_tft_model.model.eval()
                self.logger.info(f"MCE TFT model loaded successfully from {tft_model_path}.")
                self.mce_pipeline = MCE_RealtimePipeline(
                    self.mce_feature_engine, self.mce_normalizer, self.mce_tft_model
                )
                self.logger.info("MCE Realtime Pipeline initialized.")
            except Exception as e:
                self.logger.error(f"Failed to load MCE TFT model from {tft_model_path}: {e}. MCE Pipeline disabled.")
                self.mce_pipeline = None
        else:
            self.logger.warning(f"MCE TFT model not found at {tft_model_path}. MCE Pipeline disabled.")
            self.mce_pipeline = None

        self.microstructure_engine = MicrostructureFeatureEngine()
        self.latest_influence_scores = {}

        tda_window_size = 50
        tda_embedding_dim = 3
        tda_embedding_delay = 1
        tda_threshold_std = 2.0

        self.topological_radar = TopologicalRadar(
            window_size=tda_window_size,
            embedding_dim=tda_embedding_dim,
            embedding_delay=tda_embedding_delay
        )
        self.logger.info("Topological Radar initialized.")
        if not self.topological_radar.ripser_available:
            self.logger.warning("`ripser` library not found. Topological Radar will use a simplified persistence calculation. For full TDA capabilities, please install `ripser`.")

        self.persistence_history = {symbol: [] for symbol in self.config.ASSET_UNIVERSE}
        self.tda_threshold_std = tda_threshold_std

        influence_pe_window = 100
        influence_te_window = 100
        influence_gat_model_path = "models/influence_gat_model.pth"

        try:
            self.influence_mapper = InfluenceMapperPipeline(
                asset_list=self.config.ASSET_UNIVERSE,
                pe_window=influence_pe_window,
                te_window=influence_te_window
            )
            self.logger.info("Influence Mapper Pipeline initialized.")
            if not os.path.exists(influence_gat_model_path):
                 self.logger.warning(f"Influence Mapper GAT model not found at {influence_gat_model_path}. "
                                     f"Mapper will use basic TE influence. Needs training/model file for GAT.")
            else:
                self.logger.info(f"Influence Mapper GAT model found at {influence_gat_model_path}. The pipeline will attempt to use it.")

        except Exception as e:
            self.logger.error(f"Failed to initialize Influence Mapper: {e}. Influence mapping disabled.")
            self.influence_mapper = None

        # --- Refactored Initialization for Shared State ---
        # 1. Discover initial models from disk to bootstrap the Hedge Manager
        initial_model_ids = self._discover_initial_models()

        # 2. Create the single, shared HedgeEnsembleManager instance
        from forge.models.ensemble_manager import HedgeEnsembleManager
        self.hedge_manager = HedgeEnsembleManager(
            model_ids=initial_model_ids,
            learning_rate=self.config.HEDGE_ENSEMBLE_LEARNING_RATE
        )

        # Initialize Cerebellum Link
        self.cerebellum_link = CerebellumLink(self.config, fill_callback=self.handle_fill_report)

        # Load the IEL agent
        try:
            self.iel_agent = PPO.load("IEL_agent.zip")
            self.logger.info("[Crucible] IEL agent loaded successfully.")
        except Exception as e:
            self.logger.error(f"[Crucible] Failed to load IEL agent: {e}")
            self.iel_agent = None

        # 3. Initialize ArenaManager and pass the hedge_manager to it
        self.arena = ArenaManager(app_config, self.hedge_manager)
        
        # 4. Initialize SingularityEngine and pass the hedge_manager to it
        self.singularity_engine = SingularityEngine(
            trigger_trader_restart_func=self._trigger_agent_rescan,
            trigger_risk_reduction_func=self._trigger_risk_reduction,
            trigger_weight_update_func=self.update_ensemble_weights,
            status_reporter=self.pipeline_status,
            stop_event=self.stop_event,
            app_config=self.config,
            hedge_manager=self.hedge_manager, # Pass the shared instance
            logger=self.singularity_logger,
            submit_forge_task_func=self.submit_forge_task,
            shared_state=self.shared_state
        )

    def handle_fill_report(self, report_data):
        # VELOCITY V2: Gracefully handle heartbeat messages
        if "Heartbeat" in report_data:
            self.logger.debug(f"Received heartbeat: {report_data}")
            return

        strategy_id = report_data.get("strategy_id")
        # The symbol from Cerebellum is sanitized, e.g., "BTCUSDT"
        sanitized_symbol = report_data.get("symbol")
        
        if not strategy_id or not sanitized_symbol:
            self.logger.error(f"Received incomplete fill report: {report_data}")
            return

        # Find the full symbol name from the universe (e.g., "BTC/USDT")
        agent_symbol = next((s for s in self.config.ASSET_UNIVERSE if s.startswith(sanitized_symbol)), None)
        
        if agent_symbol:
            agent = self.arena.get_agent_by_id(agent_symbol, strategy_id)
            if agent:
                # Pass the report to the specific agent instance
                agent.agent.handle_fill_report(report_data)
            else:
                self.logger.error(f"Could not find agent with ID {strategy_id} for symbol {agent_symbol} to handle fill report.")
        else:
            self.logger.error(f"Could not find a symbol in ASSET_UNIVERSE matching sanitized symbol {sanitized_symbol} from fill report.")

    def _discover_initial_models(self) -> List[str]:
        """Scans the models directory to find all initial model IDs."""
        model_ids = []
        models_dir = self.config.MODEL_DIR
        if not os.path.exists(models_dir):
            return []
        for item_name in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item_name)
            if os.path.isdir(item_path):
                metadata_path = os.path.join(item_path, 'metadata.json')
                if os.path.exists(metadata_path):
                    model_ids.append(item_name)
        return model_ids


    def _setup_logger(self, name="CrucibleEngine"):
        logger = logging.getLogger(name)
        logger.propagate = False
        logger.setLevel(logging.INFO)
        
        if logger.hasHandlers():
            logger.handlers.clear()
            
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(f'%(asctime)s - [{name}] - %(levelname)s - %(message)s') 
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        return logger

    def _setup_exchange(self):
        try:
            exchange_class = getattr(ccxt, self.config.ACTIVE_EXCHANGE)
            exchange = exchange_class({'options': {'defaultType': 'spot'}})
            if self.config.ACTIVE_EXCHANGE == 'bybit':
                exchange.set_sandbox_mode(self.config.BYBIT_USE_TESTNET)
            self.logger.info(f"Basic exchange '{self.config.ACTIVE_EXCHANGE}' setup complete.")
            return exchange
        except Exception as e:
            self.logger.critical(f"Failed to setup basic exchange: {e}")
            raise

    async def _trade_watcher_and_resampler(self):
        """
        Continuously fetches live market data for all assets across all timeframes and queues for processing.
        """
        self.logger.info("Trade watcher started - monitoring live market data...")

        while not self.stop_event.is_set():
            try:
                for symbol in self.config.ASSET_UNIVERSE:
                    if self.stop_event.is_set():
                        break

                    try:
                        # Fetch data for all timeframes
                        mtfa_data = {}
                        for tf_name in self.config.TIMEFRAMES.values():
                            limit = 200 # Get enough bars for indicators
                            df = await get_market_data(symbol, tf_name, limit, exchange_instance=self.exchange)
                            if df is not None and not df.empty:
                                mtfa_data[tf_name] = df
                        
                        if mtfa_data:
                            # Queue the dictionary of dataframes
                            await self.raw_data_queue.put((symbol, mtfa_data))
                            self.logger.debug(f"Queued MTFA data for {symbol}")
                        else:
                            self.logger.warning(f"No data received for {symbol}")

                    except Exception as e:
                        self.logger.error(f"Error fetching data for {symbol}: {e}")

                await asyncio.sleep(60)

            except asyncio.CancelledError:
                self.logger.info("Trade watcher cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in trade watcher loop: {e}", exc_info=True)
                await asyncio.sleep(10)

        self.logger.info("Trade watcher stopped")

    async def _initialize_regime_models(self):
        """Initialize regime detection models for each asset on startup."""
        self.logger.info("Initializing HMM regime models for market state detection...")

        # Check if regime models already exist
        existing_regimes = {}
        for symbol in self.config.ASSET_UNIVERSE:
            sanitized = symbol.split(':')[0].replace('/', '')
            regime_path = os.path.join(self.config.MODEL_DIR, f"{sanitized}_HMM.joblib")
            if os.path.exists(regime_path):
                try:
                    from forge.strategy.predictive_regimes import PredictiveRegimeModel
                    regime_model = PredictiveRegimeModel.load(regime_path)
                    existing_regimes[symbol] = regime_model
                    self.logger.info(f"Loaded existing HMM regime model for {symbol}")
                except Exception as e:
                    self.logger.error(f"Error loading HMM regime model for {symbol}: {e}")

        if existing_regimes:
            # Use existing models
            for symbol, model in existing_regimes.items():
                if symbol not in self.arena.asset_specific_models:
                    self.arena.asset_specific_models[symbol] = {}
                self.arena.asset_specific_models[symbol]['regime_model'] = model
            self.logger.info(f"Loaded {len(existing_regimes)} existing HMM regime models")
        else:
            # Train new regime models in parallel (using the worker function already defined)
            self.logger.info("No existing HMM regime models found - training new ones...")
            if self.executor:
                tasks = []
                for symbol in self.config.ASSET_UNIVERSE:
                    task = asyncio.get_running_loop().run_in_executor(
                        self.executor,
                        _initialize_regime_worker,
                        symbol,
                        self.config.ACTIVE_EXCHANGE,
                        self.config.ACTIVE_EXCHANGE == 'bybit'
                    )
                    tasks.append(task)

                results = await asyncio.gather(*tasks, return_exceptions=True)

                for res in results:
                    if isinstance(res, tuple) and len(res) == 2:
                        symbol, regime_model = res
                        if regime_model is not None:
                            if symbol not in self.arena.asset_specific_models:
                                self.arena.asset_specific_models[symbol] = {}
                            self.arena.asset_specific_models[symbol]['regime_model'] = regime_model
                            self.logger.info(f"HMM Regime model trained for {symbol}")

        self.logger.info("HMM Regime model initialization complete")

    async def _arena_loop(self):
        """
        Main arena processing loop - takes raw data, processes features, and triggers agent decisions.
        This is where trading actually happens!
        """
        self.logger.info("Arena loop started - agents ready for battle...")

        while not self.stop_event.is_set():
            try:
                # Get raw data from queue (with timeout to allow checking stop_event)
                try:
                    symbol, mtfa_data = await asyncio.wait_for(self.raw_data_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue  # No data, loop again

                self.logger.debug(f"Processing {symbol} MTFA data through feature pipeline...")

                # Process features using the unified MTFA pipeline
                try:
                    loop = asyncio.get_running_loop()
                    
                    # Offload the entire MTFA processing to a thread pool
                    df_features = await loop.run_in_executor(
                        None,
                        process_and_align_mtfa,
                        mtfa_data,
                        self.config,
                        self.exchange
                    )

                    if df_features is None or df_features.empty:
                        self.logger.warning(f"Feature processing returned empty for {symbol}")
                        continue
                    
                    # --- The rest of the loop continues from here ---
                    # Get predicted regime
                    predicted_regime = 0
                    if self.arena.asset_specific_models.get(symbol, {}).get('regime_model'):
                        regime_model = self.arena.asset_specific_models[symbol]['regime_model']

                        # SAFETY CHECK: Verify all required features exist
                        missing_features = [col for col in regime_model.feature_columns if col not in df_features.columns]
                        if missing_features:
                            self.logger.warning(f"Regime model for {symbol} requires features {missing_features} which are missing. Available: {list(df_features.columns[:10])}... Skipping regime prediction.")
                        else:
                            features_for_regime = df_features[regime_model.feature_columns]
                            next_regime_probas = regime_model.predict_next_regime_proba(features_for_regime)
                            predicted_regime = np.argmax(next_regime_probas, axis=1)[-1]

                    # Get sentiment and strategic bias (placeholder - can be enhanced)
                    sentiment_score = 0.5  # Neutral
                    strategic_bias = 'NEUTRAL'
                    forecasted_volatility = df_features.get('atr', pd.Series([0])).iloc[-1] if 'atr' in df_features.columns else 0

                    # Get MCE signals
                    mce_skew, mce_pressure = 0.0, 0.0
                    if self.mce_pipeline:
                        try:
                            sanitized_symbol = symbol.replace('/', '')
                            book = self.l2_collector.get_book(sanitized_symbol)
                            if book and book.initialized:
                                l2_snapshot = book.get_snapshot(depth=50)
                                microstructure_features = self.microstructure_engine.calculate_all_features(l2_snapshot, sanitized_symbol)
                                
                                mce_output = self.mce_pipeline.process_update(sanitized_symbol, microstructure_features, l2_snapshot)
                                mce_skew = mce_output.get('MCE_Skew', 0.0)
                                mce_pressure = mce_output.get('MCE_Pressure', 0.0)
                                self.logger.debug(f"[{symbol}] MCE Output: Skew={mce_skew:.3f}, Pressure={mce_pressure:.3f}")
                            else:
                                self.logger.warning(f"[{symbol}] No L2 snapshot available for MCE calculation.")
                        except Exception as mce_e:
                            self.logger.error(f"Error during MCE processing for {symbol}: {mce_e}")

                    # Get KCM risk multiplier
                    prices = {symbol: df_features['close'].iloc[-1]}
                    kcm_adjustments = self.kcm.process_market_update(prices)
                    risk_multiplier = kcm_adjustments['risk_adjustments'][symbol]['risk_multiplier']

                    influence_scores = self.latest_influence_scores.get(symbol, {})
                    influence_incoming = influence_scores.get('influence_incoming', 0.0)
                    influence_outgoing = influence_scores.get('influence_outgoing', 0.0)
                    predicted_entropy_delta = influence_scores.get('predicted_entropy_delta', 0.0)

                    # TDA Regime Change Detection
                    regime_change_detected = False
                    if self.topological_radar and not df_features.empty:
                        try:
                            returns_series = df_features['close'].pct_change().dropna()
                            if len(returns_series) >= self.topological_radar.window_size:
                                recent_returns = returns_series.tail(self.topological_radar.window_size).values
                                history = self.persistence_history.get(symbol, [])
                                
                                loop = asyncio.get_running_loop()
                                regime_change_detected = await loop.run_in_executor(
                                    None,
                                    self.topological_radar.detect_regime_change,
                                    recent_returns,
                                    history,
                                    self.tda_threshold_std
                                )
                                self.persistence_history[symbol] = history

                                if regime_change_detected:
                                    self.logger.warning(f"[{symbol}] Topological Radar detected REGIME CHANGE!")
                            else:
                                self.logger.debug(f"[{symbol}] Insufficient data ({len(returns_series)}) for TDA window ({self.topological_radar.window_size}).")
                        except Exception as tda_e:
                            self.logger.error(f"Error during Topological Radar processing for {symbol}: {tda_e}")

                    # Pass to arena for agent decisions
                    await self.arena.process_data(
                        symbol=symbol,
                        data=df_features,
                        sentiment_score=sentiment_score,
                        strategic_bias=strategic_bias,
                        forecasted_volatility=forecasted_volatility,
                        risk_multiplier=risk_multiplier,
                        mce_skew=mce_skew,
                        mce_pressure=mce_pressure,
                        influence_incoming=influence_incoming,
                        influence_outgoing=influence_outgoing,
                        predicted_entropy_delta=predicted_entropy_delta,
                        regime_change_detected=regime_change_detected
                    )

                    self.logger.debug(f"✅ {symbol} processed and passed to arena agents")

                except Exception as e:
                    self.logger.error(f"Error processing features for {symbol}: {e}", exc_info=True)

            except asyncio.CancelledError:
                self.logger.info("Arena loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in arena loop: {e}", exc_info=True)
                await asyncio.sleep(5)

        self.logger.info("Arena loop stopped")

    def _execute_trade_with_iel(self, symbol: str, strategy_action: str, observation: dict):
        """Executes a trade using the Intelligent Execution Layer (IEL)."""
        if not self.iel_agent:
            self.logger.warning("[IEL] IEL agent not available. Falling back to simple market order.")
            # Fallback to simple market order
            # This part needs to be implemented based on the existing trade execution logic
            return

        # Prepare the observation for the IEL agent
        iel_obs = {
            "ofi": np.array([observation['ofi']], dtype=np.float32),
            "book_pressure": np.array([observation['book_pressure']], dtype=np.float32),
            "liquidity_density": np.array([observation['liquidity_density']], dtype=np.float32),
            "volatility": np.array([observation['volatility']], dtype=np.float32),
            "confidence": np.array([observation['confidence']], dtype=np.float32),
        }

        # Get the action from the IEL agent
        iel_action, _ = self.iel_agent.predict(iel_obs, deterministic=True)

        # Execute the trade based on the IEL agent's decision
        # This part needs to be implemented based on the exchange API
        if iel_action == 0: # Market Buy
            self.logger.info(f"[IEL] Executing Market Buy for {symbol}")
            # self.exchange.create_market_buy_order(symbol, ...)
        elif iel_action == 1: # Market Sell
            self.logger.info(f"[IEL] Executing Market Sell for {symbol}")
            # self.exchange.create_market_sell_order(symbol, ...)
        else: # Hold
            self.logger.info(f"[IEL] Holding for {symbol}")

    async def _initialize_immune_system(self):
        """Initialize the immune system for risk monitoring and anomaly detection."""
        self.logger.info("Initializing immune system for risk monitoring...")

        try:
            loop = asyncio.get_running_loop()

            # Initialize in thread pool (involves ML model loading)
            self.immune_system = await loop.run_in_executor(
                None,
                ImmuneSystem,
                100  # Placeholder for number of features
            )

            self.logger.info("Immune system initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize immune system: {e}")
            self.logger.warning("Continuing without immune system - risk monitoring disabled")
            self.immune_system = None

    async def submit_forge_task(self, symbol: str, priority: int, source: str, inherit_dna: bool = False, model_instance_id: str = None, done_callback=None):
        self.logger.info("CrucibleEngine.submit_forge_task called")
        if model_instance_id is None:
            raise ValueError("model_instance_id must be provided for Forge task submission.")

        task_id = model_instance_id

        if task_id in self.active_forge_tasks:
            self.logger.warning(f"[{source}] Forge task for {model_instance_id} is already running. Ignoring duplicate request.")
            if done_callback:
                done_callback()
            return False

        # UPDATE V5: Implement DNA inheritance from elite models
        serialized_dna = None
        if inherit_dna:
            try:
                from forge.crucible.elite_preservation import ElitePreservationSystem
                elite_system = ElitePreservationSystem(logger=self.logger)
                elite_dna = elite_system.get_best_elite_dna(
                    symbol=symbol,
                    architecture="GP2_Evolved",
                    model_instance_id=model_instance_id
                )
                if elite_dna:
                    serialized_dna = dill.dumps(elite_dna)
                    self.logger.info(f"[{source}] 🧬 Elite DNA retrieved and serialized for {model_instance_id}")
                else:
                    self.logger.info(f"[{source}] No elite DNA found for {model_instance_id}. Starting from scratch.")
            except Exception as e:
                self.logger.warning(f"[{source}] Failed to retrieve elite DNA: {e}. Starting from scratch.")

        self.logger.info(f"[{source}] Submitting Forge task (P{priority}): {symbol} (Instance: {model_instance_id}). DNA inherited: {serialized_dna is not None}")

        task = {
            'symbol': symbol,
            'model_instance_id': model_instance_id,
            'source': source,
            'inherit_dna': inherit_dna,
            'serialized_dna': serialized_dna,
            'done_callback': done_callback
        }

        # Use counter as tiebreaker to prevent dict comparison errors in PriorityQueue
        self.task_counter += 1
        queue_item = (priority, self.task_counter, task)

        self.logger.info(f"Putting task on forge queue. Queue size: {self.forge_queue.qsize()}")

        try:
            self.active_forge_tasks.add(task_id)
            asyncio.get_running_loop().create_task(self.forge_queue.put(queue_item))
            return True
        except RuntimeError:
            self.logger.error("Cannot submit Forge task: Asyncio loop is not running.")
            self.active_forge_tasks.remove(task_id)
            return False

    async def _forge_processing_loop(self):
        """
        UPDATE V5: Processes Forge tasks with unique model_instance_id tracking.
        """
        self.logger.info("Starting centralized Forge processing loop (Update V5)...")
        loop = asyncio.get_running_loop()
        while not self.stop_event.is_set():
            self.logger.info("Forge processing loop waiting for a task...")
            try:
                priority, counter, task = await self.forge_queue.get()
                self.logger.info(f"Forge processing loop got a task: {task.get('model_instance_id')}")
                symbol = task['symbol']
                model_instance_id = task['model_instance_id']
                source = task.get('source', 'Unknown')
                done_callback = task.get('done_callback')
                serialized_dna = task.get('serialized_dna')
                inferred_opponent_params = task.get('inferred_opponent_params')

                task_id = model_instance_id
                dna_status = "with inherited DNA" if serialized_dna else "from scratch"
                self.logger.info(f"--- Processing Forge task from {source} (P{priority}): {symbol} (Instance: {model_instance_id}) {dna_status} ---")

                try:
                    if self.executor:
                        # UPDATE V5: Pass generation (can extract from model_instance_id or use 0 for now)
                        generation = 0  # Will be properly tracked in future versions
                        try:
                            result = await asyncio.wait_for(loop.run_in_executor(
                                self.executor,
                                run_forge_process,
                                symbol, model_instance_id, generation, serialized_dna, self.shared_state
                            ), timeout=7200) # 2 hour timeout
                        except asyncio.TimeoutError:
                            self.logger.critical(f"Forge task for {model_instance_id} timed out after 2 hours.")
                            result = None
                        
                        if result and isinstance(result, tuple) and len(result) == 3:
                            returned_symbol, new_agent_id, serialized_winning_dna = result
                            
                            if new_agent_id:
                                new_dna = None
                                if serialized_winning_dna:
                                    try:
                                        from deap import base, creator, gp as deap_gp
                                        if not hasattr(creator, "FitnessMax"):
                                            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
                                        if not hasattr(creator, "Individual"):
                                            creator.create("Individual", deap_gp.PrimitiveTree, fitness=creator.FitnessMax)

                                        new_dna = dill.loads(serialized_winning_dna)
                                    except Exception as e:
                                        self.logger.error(f"Failed to deserialize winning DNA: {e}")

                                self.pit_crew.add_challenger(returned_symbol, new_agent_id, 0) # Generation is not tracked anymore
                                self.logger.info(f"[VELOCITY] Model {new_agent_id} added to Challenger Bench for {returned_symbol}")

                                from forge.core.agent import V3Agent
                                from forge.crucible.arena_manager import CrucibleAgent
                                new_v3_agent = V3Agent(symbol=returned_symbol, exchange=self.exchange, app_config=self.config, agent_id=new_agent_id)
                                new_crucible_agent = CrucibleAgent(agent_id=new_agent_id, v3_agent=new_v3_agent, dna=new_dna)
                                await self.arena.add_agent(returned_symbol, new_crucible_agent)

                                try:
                                    new_v3_agent.specialist_models = new_v3_agent._load_specialist_models()
                                    self.logger.info(f"✅ AGENT {new_agent_id} IS LIVE AND ARMED. Models loaded: {len(new_v3_agent.specialist_models)}. DNA captured: {new_dna is not None}.")
                                except Exception as e:
                                    self.logger.error(f"❌ Failed to load models for agent {new_agent_id}: {e}")
                            else:
                                self.logger.info(f"Forge task {task_id} completed, but no new model was registered.")
                        else:
                            self.logger.error(f"Forge task {task_id} failed or returned an invalid result.")
                except Exception as e:
                    self.logger.error(f"Error during execution of Forge task {task_id}: {e}", exc_info=True)
                finally:
                    self.forge_queue.task_done()
                    if task_id in self.active_forge_tasks:
                        self.active_forge_tasks.remove(task_id)
                    if done_callback:
                        done_callback()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in Forge processing loop: {e}", exc_info=True)
                await asyncio.sleep(10)

    async def run(self):
        self.loop = asyncio.get_running_loop()
        self.logger.info("Starting Hyper-Evolutionary Engine...")

        if self.is_main_instance and self.executor is None:
            try:
                mp_context = multiprocessing.get_context() 
                max_workers = max(1, os.cpu_count() - 1)
                self.executor = ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context)
                self.logger.info(f"ProcessPoolExecutor initialized in run() using '{mp_context.get_start_method()}' context with {max_workers} workers.")
            except Exception as e:
                self.logger.error(f"Failed to initialize ProcessPoolExecutor in run(). Error: {e}")
                return

        self.logger.info("--- Phase 1: Initializing Core Systems (Waiting for completion)... ---")
        self.l2_collector.start()
        self.logger.info("L2 Collector thread started.")
        await self._initialize_regime_models()
        await self._initialize_immune_system()
        await self.load_existing_models() # Load all existing models into agents
        self.logger.info("--- Phase 1: Complete. ---")

        self.logger.info("--- Phase 2: Starting Main Operational Loops (Concurrent)... ---")
        
        # Phase 2: Start concurrent operational loops
        await asyncio.gather(
            self._forge_processing_loop(),
            self.pit_crew.shadow_evaluation_loop(self.arena, self.stop_event),
            self._trade_watcher_and_resampler(),
            self._arena_loop(),
            self._watchtower_loop(),
            self.singularity_engine.run(),
            self._influence_mapper_loop()
        )

    async def load_existing_models(self):
        """
        UPDATE V5: Load existing models as separate agents per model instance.
        Creates individual agents for each model found on disk.
        """
        self.logger.info("Scanning for and loading all existing models from model directories...")
        models_dir = self.config.MODEL_DIR
        if not os.path.exists(models_dir):
            self.logger.warning("Models directory not found. No agents will be loaded.")
            return

        # Scan for all model directories
        loaded_count = 0
        for item_name in os.listdir(models_dir):
            if "_unified" in item_name:
                continue
            item_path = os.path.join(models_dir, item_name)
            if os.path.isdir(item_path):
                metadata_path = os.path.join(item_path, 'metadata.json')
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        symbol = metadata.get("asset_symbol")

                        # Skip if symbol not in asset universe
                        if symbol not in self.config.ASSET_UNIVERSE:
                            self.logger.warning(f"Model {item_name} is for {symbol}, which is not in ASSET_UNIVERSE. Skipping.")
                            continue

                        # UPDATE V5: Use the model directory name as the unique model_instance_id
                        model_instance_id = item_name

                        # Check if agent with this ID already exists
                        if self.arena.get_agent_by_id(symbol, model_instance_id):
                            self.logger.info(f"Agent {model_instance_id} already exists. Skipping.")
                            continue

                        self.logger.info(f"Loading model instance: {model_instance_id} for {symbol}")

                        # Create a separate V3Agent for this model instance
                        from forge.core.agent import V3Agent
                        from forge.crucible.arena_manager import CrucibleAgent

                        new_v3_agent = V3Agent(
                            symbol=symbol,
                            exchange=self.exchange,
                            app_config=self.config,
                            agent_id=model_instance_id,
                            iel_agent=self.iel_agent,
                            cerebellum_link=self.cerebellum_link
                        )
                        new_crucible_agent = CrucibleAgent(
                            agent_id=model_instance_id,
                            v3_agent=new_v3_agent,
                            dna=None
                        )

                        # Add agent to arena
                        await self.arena.add_agent(symbol, new_crucible_agent)

                        # Load specialist models for this agent
                        new_v3_agent.specialist_models = new_v3_agent._load_specialist_models()
                        loaded_count += 1
                        self.logger.info(f"✅ AGENT {model_instance_id} for {symbol} is LIVE. Loaded {len(new_v3_agent.specialist_models)} specialist model(s).")

                    except Exception as e:
                        self.logger.error(f"Error loading model {item_name}: {e}", exc_info=True)

        self.logger.info(f"--- Finished loading existing agents. Total loaded: {loaded_count} ---")

    async def load_agents_from_disk(self):
        """Public method to trigger a re-scan and load of agents."""
        await self.load_existing_models()

    async def load_agent(self, model_id: str):
        """Creates and loads a single agent based on a model_id."""
        self.logger.info(f"Attempting to dynamically load agent for model: {model_id}")
        
        # Basic parsing from model_id, assuming format like 'BTCUSDT_...'
        parts = model_id.split('_')
        asset_symbol_str = parts[0] if parts else None
        if not asset_symbol_str:
            self.logger.error(f"Could not determine asset symbol from model_id: {model_id}")
            return

        # Find the full symbol name from the universe
        symbol = next((s for s in self.config.ASSET_UNIVERSE if s.startswith(asset_symbol_str)), None)
        if not symbol:
            self.logger.error(f"Asset symbol '{asset_symbol_str}' from model_id not found in ASSET_UNIVERSE.")
            return

        # Check if an agent for this symbol already exists
        if self.arena.get_agents_for_symbol(symbol):
            self.logger.info(f"Agent for {symbol} already exists. New model will be loaded on next rescan.")
            # Here, you might want to just trigger a rescan for the existing agent
            # For now, we'll let the regular process handle it.
            return

        self.logger.info(f"Creating a new V3Agent for {symbol} to manage model {model_id}.")
        agent_id = f"{asset_symbol_str}_ensemble_agent"

        try:
            # Create and add the agent
            new_v3_agent = V3Agent(symbol=symbol, exchange=self.exchange, app_config=self.config, agent_id=agent_id, iel_agent=self.iel_agent, cerebellum_link=self.cerebellum_link)
            new_crucible_agent = CrucibleAgent(agent_id=agent_id, v3_agent=new_v3_agent, dna=None)
            await self.arena.add_agent(symbol, new_crucible_agent)

            # Load the models for the new agent
            await new_v3_agent._reload_models()
            self.logger.info(f"✅ DYNAMIC LOAD: Agent {agent_id} for {symbol} is LIVE. Loaded {len(new_v3_agent.specialist_models)} models.")

        except Exception as e:
            self.logger.error(f"Failed to create or load agent for {symbol}: {e}", exc_info=True)

    def _trigger_agent_rescan(self, symbol: str):
        # ... (Implementation remains the same)
        pass

    def _trigger_risk_reduction(self):
        # ... (Implementation remains the same)
        pass

    def update_ensemble_weights(self, losses: dict):
        if self.hedge_manager:
            self.hedge_manager.update_weights(losses)
            self.arena.broadcast_hedge_weights(self.hedge_manager.get_weights())

    async def _watchtower_loop(self):
        while not self.stop_event.is_set():
            await asyncio.sleep(900) # 15 minutes
            self.logger.info("Watchtower commencing judgment...")
            
            try:
                with open("performance_log.jsonl", "r") as f:
                    trade_logs = [json.loads(line) for line in f]
            except (FileNotFoundError, json.JSONDecodeError):
                trade_logs = []

            if not trade_logs:
                self.logger.info("No trades logged yet. Skipping judgment.")
                continue

            try:
                agents_to_replace = self.watchtower.judge_agents(self.arena.agents, trade_logs)
                if agents_to_replace:
                    self.logger.info(f"Watchtower identified {len(agents_to_replace)} agents for replacement.")
                    for symbol, terminated_agent_id in agents_to_replace:

                        # UPDATE V5: Watchtower reaping now triggers DNA inheritance
                        # The elite DNA will be automatically retrieved based on model_instance_id
                        self.logger.warning(f"REAPING Agent {terminated_agent_id}. Elite DNA will be inherited for replacement.")
                        await self.arena.remove_agent(symbol, terminated_agent_id)

                        # UPDATE V5: Use the terminated agent ID as the model_instance_id for replacement
                        # This ensures the new model inherits from the same lineage
                        model_instance_id = terminated_agent_id

                        await self.submit_forge_task(
                            symbol=symbol,
                            priority=PRIORITY_HIGH,
                            source="Watchtower",
                            inherit_dna=True,  # CRITICAL: Enable DNA inheritance when reaping
                            model_instance_id=model_instance_id
                        )
            except Exception as e:
                self.logger.error(f"Error during Watchtower judgment: {e}", exc_info=True)

            self.logger.info("Watchtower judgment complete.")

    async def stop(self):
        self.logger.info("CrucibleEngine stopping...")
        self.stop_event.set() # Signal all async loops to stop

        # Close the main exchange connection used for data fetching
        if self.exchange:
            try:
                await self.exchange.close()
                self.logger.info("Main exchange connection closed.")
            except Exception as e:
                self.logger.error(f"Error closing main exchange connection: {e}")

        # Shutdown the process pool executor gracefully
        if self.executor:
            self.logger.info("Shutting down ProcessPoolExecutor...")
            # Run shutdown in a separate thread as it's blocking
            await asyncio.get_running_loop().run_in_executor(
                None, self.executor.shutdown, True # wait=True
            )
            self.logger.info("ProcessPoolExecutor shut down.")

        # Add any other necessary cleanup (e.g., closing CerebellumLink sockets if needed)
        # if self.cerebellum_link:
        #     # self.cerebellum_link.close() # Assuming a close method exists

        if hasattr(self, 'l2_collector') and self.l2_collector:
            try:
                self.l2_collector.stop()
                self.logger.info("L2 Collector stopped.")
            except Exception as e:
                self.logger.error(f"Error stopping L2 Collector: {e}")

        self.logger.info("CrucibleEngine stopped.")

    async def _influence_mapper_loop(self):
        """Periodically updates the Influence Mapper."""
        while not self.stop_event.is_set():
            await asyncio.sleep(60) # Update every 60 seconds
            if self.influence_mapper:
                try:
                    tickers = await self.exchange.fetch_tickers(self.config.ASSET_UNIVERSE)
                    latest_prices = {symbol: ticker['last'] for symbol, ticker in tickers.items()}

                    if len(latest_prices) == len(self.config.ASSET_UNIVERSE):
                        influence_output = await asyncio.get_running_loop().run_in_executor(
                            None, self.influence_mapper.process_update, latest_prices
                        )
                        self.latest_influence_scores = influence_output
                        self.logger.debug("Influence Mapper updated.")
                    else:
                         self.logger.debug("Influence Mapper waiting for complete market price data.")

                except Exception as e:
                    self.logger.error(f"Error in Influence Mapper loop: {e}")