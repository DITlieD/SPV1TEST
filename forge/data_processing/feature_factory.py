import pandas as pd
import numpy as np
import pandas_ta as ta
# VELOCITY UPGRADE: Replace CEEMDAN with VMD for 10-50x speedup
from vmdpy import VMD
import pywt  # Wavelet transform as backup
import logging
import torch
import asyncio # Add missing import
import warnings
from forge.strategy.particle_filter import MarketStateTracker
from data_processing_v2 import run_graph_feature_pipeline
from forge.evolution.feature_synthesizer import evolve_features
from forge.data_processing.order_flow_engine import calculate_order_flow_imbalance, calculate_book_pressure, calculate_liquidity_density
from data_fetcher import get_l2_order_book
import os

class FeatureFactory:
    def __init__(self, data_universe: dict, logger=None):
        self.data_universe = {symbol: df.copy() for symbol, df in data_universe.items()}
        self.logger = logger if logger else logging.getLogger(__name__)
        self.log = self.logger.info

    def _process_single_asset(self, df: pd.DataFrame) -> pd.DataFrame:
        symbol = df.attrs.get('symbol', 'unknown')

        if 'close' not in df.columns:
            self.log(f"  -> [{symbol}] WARNING: 'close' column not found. Skipping.")
            return df

        ohlcv_cols = df[['open', 'high', 'low', 'close', 'volume']].copy()
        
        # --- DIAGNOSTIC LOGGING ---
        self.log(f"  -> [DIAGNOSTIC] [{symbol}] 1. Calculating Technical Indicators...")
        df = self._calculate_technical_indicators(df)
        
        self.log(f"  -> [DIAGNOSTIC] [{symbol}] 2. Generating VMD Features (Velocity Upgrade - 10-50x faster)...")
        df = self._generate_vmd_features(df)
        self.log(f"  -> [DIAGNOSTIC] [{symbol}] 2. VMD Complete.")
        
        self.log(f"  -> [DIAGNOSTIC] [{symbol}] 3. Generating Particle Filter Features (Potential Hang Point)...")
        df = self._generate_particle_filter_features(df)
        self.log(f"  -> [DIAGNOSTIC] [{symbol}] 3. Particle Filter Complete.")
        
        df = self._generate_order_flow_features(df)
        self.log(f"  -> [DIAGNOSTIC] [{symbol}] 4. Order Flow Features Complete.")
        
        # CRITICAL FIX: Define numeric_df BEFORE sanitization
        numeric_df = df.select_dtypes(include=np.number).copy()
        
        # --- FINAL SANITIZATION (CRITICAL FIX) ---
        numeric_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        interpolation_method = 'time' if isinstance(df.index, pd.DatetimeIndex) else 'linear'
        try:
            numeric_df.interpolate(method=interpolation_method, limit_direction='both', axis=0, inplace=True)
        except Exception as e:
            self.log(f"  -> WARNING: Interpolation failed. Error: {e}. Falling back.")

        numeric_df.fillna(method='ffill', inplace=True)
        numeric_df.fillna(method='bfill', inplace=True)
        numeric_df.fillna(0, inplace=True) # Absolute fallback
        # ---------------------------------------

        final_df = pd.concat([ohlcv_cols, numeric_df], axis=1)
        return final_df.loc[:,~final_df.columns.duplicated()]

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # self.log(f"  -> Calculating technical indicators for {df.attrs.get('symbol', 'unknown')}...")
        df.ta.rsi(length=14, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.sma(length=20, append=True)
        df.ta.sma(length=50, append=True)

        # Create ATRr (ATR ratio = ATR / close) for compatibility with old models
        if 'ATR_14' in df.columns:
            df['ATRr_14'] = df['ATR_14'] / (df['close'] + 1e-9)

        return df

    def _generate_vmd_features(self, df: pd.DataFrame, n_modes=6) -> pd.DataFrame:
        """
        VELOCITY UPGRADE: Variational Mode Decomposition (VMD)
        10-50x faster than CEEMDAN with better noise resistance.
        """
        signal = df['close'].values

        # Pre-validate signal
        if len(signal) < 10:
            self.log(f"  -> [{df.attrs.get('symbol', 'unknown')}] WARNING: Signal too short for VMD ({len(signal)} points). Filling with zeros.")
            for i in range(n_modes):
                df[f'vmd_{i}'] = 0
            return df

        # Clean signal
        if np.any(~np.isfinite(signal)):
            signal = pd.Series(signal).fillna(method='ffill').fillna(method='bfill').fillna(0).values
            signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

        # Check for zero variance
        if np.std(signal) < 1e-10:
            for i in range(n_modes):
                df[f'vmd_{i}'] = 0
            return df

        try:
            # VMD parameters (optimized for speed)
            alpha = 2000       # Moderate bandwidth constraint
            tau = 0            # No noise-tolerance (clean data)
            K = n_modes        # Number of modes
            DC = 0             # No DC part
            init = 1           # Initialize omegas uniformly
            tol = 1e-7         # Tolerance

            # Run VMD (significantly faster than CEEMDAN)
            modes, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)

            # VMD returns modes as rows, transpose to get each mode as column
            # modes shape: (K, signal_length)
            for i in range(min(K, n_modes)):
                if i < modes.shape[0] and modes[i].shape[0] == len(df):
                    # Clean any NaN/Inf values in the mode
                    mode_clean = np.nan_to_num(modes[i], nan=0.0, posinf=0.0, neginf=0.0)
                    df[f'vmd_{i}'] = mode_clean
                else:
                    df[f'vmd_{i}'] = 0

            # Ensure exactly n_modes features exist
            for i in range(modes.shape[0], n_modes):
                df[f'vmd_{i}'] = 0

        except Exception as e:
            self.log(f"  -> [{df.attrs.get('symbol', 'unknown')}] ERROR in VMD: {e}. Filling with zeros.")
            for i in range(n_modes):
                df[f'vmd_{i}'] = 0

        return df

    def _generate_particle_filter_features(self, df: pd.DataFrame) -> pd.DataFrame:
        from forge.strategy.particle_filter import MarketStateTracker

        atr_col = next((c for c in df.columns if c.startswith('ATRr_') or c.startswith('ATR_')), None)
        
        if not atr_col:
            self.log(f"  -> [{df.attrs.get('symbol', 'unknown')}] WARNING: ATR column not found for Particle Filter. Skipping.")
            return df

        process_uncertainty = df[atr_col] / (df['close'] + 1e-9) # Add epsilon for stability
        process_uncertainty.clip(0, 1, inplace=True) # Clip to a reasonable range (0% to 100% uncertainty)
        
        process_uncertainty.replace([np.inf, -np.inf], 0, inplace=True)
        process_uncertainty.fillna(method='bfill', inplace=True)
        process_uncertainty.fillna(method='ffill', inplace=True) 

        returns = df['close'].pct_change().fillna(0).values
        returns[np.isinf(returns)] = 0
        
        pf = MarketStateTracker()
        denoised_states, state_uncertainties = [], []
        for i in range(len(returns)):
            denoised_state = pf.step(returns[i], process_uncertainty.iloc[i])
            denoised_states.append(denoised_state[0])
            state_uncertainties.append(pf.get_state_uncertainty())
        
        df['denoised_close_pct_change'] = pd.Series(denoised_states, index=df.index)
        df['state_uncertainty'] = pd.Series(state_uncertainties, index=df.index)
        
        start_close = df['close'].bfill().iloc[0] if not df['close'].bfill().empty else np.nan
        
        if np.isfinite(start_close):
             df['denoised_close'] = start_close * (1 + df['denoised_close_pct_change']).cumprod()
        else:
             df['denoised_close'] = df['close']

        return df

    def _generate_order_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        symbol = df.attrs.get('symbol', 'unknown')
        
        if 'MTFA_TEMP' in symbol:
            self.log(f"  -> [DIAGNOSTIC] [{symbol}] Skipping Order Flow Features for temporary MTFA symbol.")
            df['ofi'] = 0.0
            df['book_pressure'] = 0.0
            df['liquidity_density'] = 0.0
            return df

        self.log(f"  -> [DIAGNOSTIC] [{symbol}] 4. Generating Order Flow Features...")

        # Fetch order book data
        order_book = asyncio.run(get_l2_order_book(symbol))

        if order_book:
            df['ofi'] = calculate_order_flow_imbalance(order_book)
            df['book_pressure'] = calculate_book_pressure(order_book)
            df['liquidity_density'] = calculate_liquidity_density(order_book)
        else:
            df['ofi'] = 0.0
            df['book_pressure'] = 0.0
            df['liquidity_density'] = 0.0
        
        return df

    async def _generate_gnn_features(self, app_config, exchange) -> dict:
        self.log("  -> Generating GNN features for the entire market...")
        gnn_features_df = await run_graph_feature_pipeline(app_config, exchange)
        if gnn_features_df.empty:
            self.log("  -> GNN feature generation returned no data. Skipping merge.")
            return self.data_universe
        updated_universe = {}
        for symbol, df in self.data_universe.items():
            if symbol in gnn_features_df.index.get_level_values('symbol'):
                asset_gnn_features = gnn_features_df.xs(symbol, level='symbol').reset_index()

                # CRITICAL FIX: Only merge GNN columns (gat_*) to avoid _x/_y suffix collision
                gnn_cols = [col for col in asset_gnn_features.columns if col.startswith('gat_')]
                asset_gnn_only = asset_gnn_features[['timestamp'] + gnn_cols]

                merged_df = pd.merge(df.reset_index(), asset_gnn_only, on='timestamp', how='left').set_index('timestamp')
                merged_df[gnn_cols] = merged_df[gnn_cols].ffill().bfill()
                updated_universe[symbol] = df if merged_df.empty else merged_df
            else:
                updated_universe[symbol] = df
        return updated_universe

    def add_evolved_features(self, evolved_programs: list):
        """Adds features evolved by the feature synthesizer to the data universe."""
        if not evolved_programs:
            self.log("--- Feature Factory: No evolved features to add. ---")
            return

        self.log(f"--- Feature Factory: Injecting {len(evolved_programs)} evolved features... ---")
        for symbol, df in self.data_universe.items():
            for i, program in enumerate(evolved_programs):
                try:
                    feature_name = f"evolved_{i}"
                    feature_values = program.execute(df.values)
                    df[feature_name] = feature_values
                except Exception as e:
                    self.log(f"  -> [{symbol}] ERROR applying evolved feature {i}: {e}")
            self.data_universe[symbol] = df

    def create_all_features(self, app_config, exchange, skip_feature_evolution=False) -> dict:
        """
        Generates features. Skips GNN if running inside the Forge worker (indicated by exchange=None).

        Args:
            app_config: Configuration object
            exchange: Exchange object (None in Forge worker context)
            skip_feature_evolution: If True, skips Feature Synthesizer (God Evolution) for faster processing
                                   Used during regime model initialization to avoid redundant evolution
        """

        self.log("--- Feature Factory: Starting Processing ---")

        # 1. Process single asset features
        for symbol, df in self.data_universe.items():
            df.attrs['symbol'] = symbol
            self.data_universe[symbol] = self._process_single_asset(df)

        # 2. Evolve new features (God Protocol - Pillar II)
        if skip_feature_evolution:
            self.log("--- Feature Factory: Skipping Feature Synthesis (disabled for this call) ---")
        else:
            self.log("--- Feature Factory: Starting Autonomous Feature Synthesis (God Evolution)... ---")
            for symbol, df in self.data_universe.items():
                if df.empty:
                    continue

                # Define target for feature evolution (e.g., future returns)
                target = df['close'].pct_change().shift(-1).fillna(0)

                # Evolve features
                evolved_df = evolve_features(df, target)

                # Add evolved features to the dataframe
                if not evolved_df.empty:
                    self.log(f"  -> [{symbol}] Injecting {len(evolved_df.columns)} evolved features...")
                    df = pd.concat([df, evolved_df], axis=1)

        # 3. Generate GNN features (Conditional Execution)
        
        if exchange is None:
            # This path is taken when called from forge_worker.py
            self.log("--- Feature Factory: Exchange object is None. Skipping GNN feature generation (Forge context). ---")
        else:
            # This path is taken if called from the main process 
            self.log("--- Feature Factory: Exchange object present. Attempting GNN generation (Main context). ---")
            
            # Handle asyncio loop management required to run an async function from a sync context
            try:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Run the async GNN pipeline and update the universe
                # We must use loop.run_until_complete to execute the async method
                updated_universe = loop.run_until_complete(self._generate_gnn_features(app_config, exchange))
                self.data_universe = updated_universe
                
            except Exception as e:
                self.log(f"ERROR during GNN feature generation: {e}. Proceeding without GNN features.")


        self.log("--- Feature Factory: All Processing Complete ---")
        return self.data_universe
