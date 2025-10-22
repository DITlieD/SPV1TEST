# train_aux_models.py (V5.4 - UADE Fully Integrated)
import os
import joblib
import pandas as pd
import numpy as np
import warnings
import multiprocessing
import time
from datetime import datetime, timedelta
import asyncio

from arch.utility.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='stable_baselines3')
warnings.filterwarnings("ignore", category=RuntimeWarning, module='numpy')

from config import ASSET_UNIVERSE, MODEL_DIR, TIMEFRAMES
from data_fetcher import get_market_data
from data_processing_v2 import add_all_features
from forge.strategy.predictive_regimes import PredictiveRegimeModel
from forge.strategy.dynamic_regimes import DynamicRegimeModel

# ... (rest of the imports)

def train_and_save_hdbscan(df: pd.DataFrame, symbol: str):
    """Trains a HDBSCAN model and saves it."""
    # ... (existing code for HDBSCAN)

def train_and_save_hmm(df: pd.DataFrame, symbol: str):
    """Trains a Gaussian HMM model and saves it."""
    sanitized_symbol = symbol.split(':')[0].replace('/', '')
    filepath = os.path.join(config.MODEL_DIR, f"{sanitized_symbol}_HMM.joblib")

    features = df.select_dtypes(include=np.number).drop(['open', 'high', 'low', 'close', 'volume'], axis=1, errors='ignore')
    
    hmm_model = PredictiveRegimeModel(n_components=4)
    hmm_model.fit(features)
    hmm_model.save(filepath)
    print(f"[HMM Trainer] Saved HMM model for {symbol} to {filepath}")

# ... (rest of the file)

async def main():
    # ... (existing code)

    # Train HMM models
    for symbol, df in all_data.items():
        if not df.empty:
            train_and_save_hmm(df, symbol)

    # ... (existing code)

from rl_governor import RLGovernor, RLGovernorEnv
from forge.modeling.volatility_engine import VolatilityEngine
from forge.data_processing.labeling import create_categorical_labels

def process_asset(asset: str):
    """
    The complete training pipeline for a single asset.
    This function is designed to be run in a separate process.
    Returns the trained models to be saved by the main process.
    """
    try:
        print(f"[Genesis] --- Processing for {asset} ---")
        
        # --- Model Freshness Check ---
        sanitized_symbol = asset.split(':')[0].replace('/', '')
        hdbscan_path = os.path.join(MODEL_DIR, f"{sanitized_symbol}_HDBSCAN.joblib")
        rlg_path = os.path.join(MODEL_DIR, f"{sanitized_symbol}_RLG.joblib")

        if os.path.exists(hdbscan_path) and os.path.exists(rlg_path):
            hdbscan_age = time.time() - os.path.getmtime(hdbscan_path)
            rlg_age = time.time() - os.path.getmtime(rlg_path)
            if hdbscan_age < timedelta(days=7).total_seconds() and rlg_age < timedelta(days=7).total_seconds():
                print(f"[Genesis] ({asset}) ✅ Auxiliary models are up-to-date. Skipping.")
                return None

        # --- Data Loading ---
        timeframe = TIMEFRAMES['strategic']
        data_path = os.path.join('data', f"{sanitized_symbol}_{timeframe}_raw.csv")

        if os.path.exists(data_path):
            df_features_real = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
        else:
            print(f"[Genesis] ({asset}) No local data found at {data_path}. Fetching from network...")
            df_features_real = asyncio.run(get_market_data(asset, timeframe, limit=5000))
            if df_features_real.empty:
                print(f"[Genesis] ({asset}) ERROR: Could not fetch data for {asset}. Skipping.")
                return None
            df_features_real.to_csv(data_path)
        
        if (df_features_real[['open', 'high', 'low', 'close']] <= 0).any().any():
            print(f"[Genesis] ({asset}) ERROR: Zero or negative price found. Skipping.")
            return None
        
        df_features_real = add_all_features(df_features_real, asset)
        
        # --- Sanitization ---
        df_features_real.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_features_real.ffill(inplace=True)
        df_features_real.bfill(inplace=True)
        df_features_real.dropna(inplace=True)

        if df_features_real.empty:
            print(f"[Genesis] ({asset}) ERROR: Data empty after sanitization. Skipping.")
            return None

        # --- HDBSCAN Training ---
        print(f"[Genesis] ({asset}) Training HDBSCAN regime detector...")
        hdbscan_model = DynamicRegimeModel()
        hdbscan_model.fit(df_features_real)

        # --- RL Governor Training ---
        print(f"[Genesis] ({asset}) Generating inputs for RL Governor...")
        vol_engine = VolatilityEngine()
        forecasts = [0.05] * 500
        for i in range(500, len(df_features_real)):
            df_window = df_features_real.iloc[i-500:i]
            forecasts.append(vol_engine.get_forecast(df_window))
        df_features_real['forecasted_vol'] = forecasts
        df_features_real.dropna(inplace=True)

        if df_features_real.empty:
            print(f"[Genesis] ({asset}) ERROR: Data empty after volatility calc. Skipping RL.")
            return None

        regime_labels = hdbscan_model.predict(df_features_real)
        num_regimes = 10
        real_regimes = np.zeros((len(df_features_real), num_regimes))
        valid_indices = (regime_labels >= 0) & (regime_labels < num_regimes)
        if np.any(valid_indices):
            real_regimes[np.arange(len(df_features_real))[valid_indices], regime_labels[valid_indices]] = 1

        p_meta_signals = pd.Series(0.5, index=df_features_real.index)
        
        print(f"[Genesis] ({asset}) Training RL Governor...")
        rl_env = RLGovernorEnv(df_features=df_features_real, p_meta_signals=p_meta_signals, hmm_regime_probs=real_regimes, ensemble_weights=None, drift_signal=0.0)
        
        # Use the global device setting from the config file
        from config import DEVICE
        rl_governor = RLGovernor(rl_env, device=DEVICE)
        
        rl_governor.train(total_timesteps=10000)
        
        return asset, hdbscan_model, rl_governor
        
    except Exception as e:
        print(f"💥 ERROR processing {asset}: {e}")
        return None

if __name__ == "__main__":
    # Set the start method to 'spawn' for CUDA compatibility on Windows/macOS
    multiprocessing.set_start_method('spawn', force=True)
    print("\n[Genesis] --- Starting Parallel Auxiliary Model Training ---")
    os.makedirs(MODEL_DIR, exist_ok=True)
    num_processes = min(multiprocessing.cpu_count(), len(ASSET_UNIVERSE))
    print(f"[Genesis] Using {num_processes} parallel processes.")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_asset, ASSET_UNIVERSE)

    print("\n[Genesis] --- Saving Trained Auxiliary Models ---")
    for result in results:
        if result:
            asset, hdbscan_model, rl_governor = result
            asset_id = asset.split(':')[0].replace('/', '')
            hdbscan_path = os.path.join(MODEL_DIR, f"{asset_id}_HDBSCAN.joblib")
            rlg_path = os.path.join(MODEL_DIR, f"{asset_id}_RLG.joblib")

            print(f"[Genesis] ({asset}) Saving HDBSCAN model...")
            joblib.dump(hdbscan_model, hdbscan_path)

            print(f"[Genesis] ({asset}) Saving RL Governor model...")
            rl_governor.save(rlg_path)

    print("\n[Genesis] --- Auxiliary Model Training Complete ---")
