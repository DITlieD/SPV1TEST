import os
import joblib
import pandas as pd
from multiprocessing import Pool, cpu_count
import time
from datetime import datetime, timedelta
import asyncio

from config import ASSET_UNIVERSE, MODEL_DIR
from data_fetcher import get_market_data
from data_processing_v2 import add_all_features, get_specialist_labels
from models_v2 import RiverOnlineModel

def process_asset_online(asset_info: tuple):
    """
    The complete pre-training pipeline for a single asset's online model.
    Designed to be run in a separate process.
    """
    symbol, timeframe, model_dir = asset_info
    try:
        print(f"\n[Genesis] --- Genesis Online Model Trainer for {symbol} ---")
        
        # --- Model Freshness Check ---
        sanitized_symbol = symbol.split(':')[0].replace('/', '')
        model_id = f"{sanitized_symbol}_ONLINE"
        model_path = os.path.join(model_dir, f"{model_id}.joblib")
        if os.path.exists(model_path):
            model_age = time.time() - os.path.getmtime(model_path)
            if model_age < timedelta(days=7).total_seconds():
                print(f"  -> [Genesis] ({symbol}) ✅ Online model is up-to-date. Skipping.")
                return

        # --- Data Loading ---
        data_path = os.path.join('data', f"{sanitized_symbol}_{timeframe}_raw.csv")

        if not os.path.exists(data_path):
            print(f"  -> [Genesis] ({symbol}) CRITICAL ERROR: Data file not found at {data_path}. Please run prepare_all_data.py first.")
            return

        df_full = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)

        # --- Feature Engineering & Labeling ---
        df_features = add_all_features(df_full.copy(), symbol)
        atr_col = next((c for c in df_features.columns if c.startswith('ATRr')), 'ATRr_14')
        df_features['label'] = get_specialist_labels(df_features, atr_col, module_type='reversion')
        df_features.dropna(inplace=True)
        if df_features.empty:
            print(f"  -> [Genesis] Skipping {symbol}: Not enough data after processing.")
            return

        # --- Model Training & Saving ---
        print(f"[Genesis] ({symbol}) Pre-training online model...")
        online_model = RiverOnlineModel()
        online_model.fit(df_features)
        
        # Save only the internal River pipeline, not the wrapper class
        joblib.dump(online_model.model, model_path)
        print(f"  -> [Genesis] ✅ Saved initial online model to {model_path}")
    except Exception as e:
        print(f"💥 ERROR processing online model for {symbol}: {e}")

if __name__ == '__main__':
    import multiprocessing as mp

    # Force spawn context for stability
    try:
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
            print("[Multiprocessing] Start method forced to 'spawn'.")
    except RuntimeError:
        print("[Multiprocessing] Warning: Start method already set.")
        pass # Ignore if already set

    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Create a list of arguments for each process
    tasks = [(asset, '15m', MODEL_DIR) for asset in ASSET_UNIVERSE]
    
    num_processes = min(cpu_count(), len(tasks))
    print(f"\n[Genesis] --- Starting Parallel Online Model Training ({num_processes} processes) ---")

    with Pool(processes=num_processes) as pool:
        pool.map(process_asset_online, tasks)

    print("\n[Genesis] --- Initial training for all online models complete. ---")
