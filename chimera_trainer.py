import os
import joblib
import pandas as pd
from multiprocessing import Pool, cpu_count
import argparse
import time
from datetime import datetime, timedelta
import numpy as np
import asyncio
import json

from config import ASSET_UNIVERSE
from data_processing_v2 import get_market_data, add_all_features, get_specialist_labels
# Correctly import the new standardized model wrapper
from models_v2 import LGBMWrapper, XGBWrapper 

def train_initial_model(asset_info: tuple):
    """
    Trains and saves a single specialist model.
    Designed to be run in a separate process.
    """
    symbol, timeframe, model_dir, model_type = asset_info
    try:
        print(f"\n[Genesis] --- Genesis Specialist Model Trainer for {symbol} ---")

        # --- Model Freshness Check ---
        base_symbol = symbol.split(':')[0]
        sanitized_symbol = base_symbol.replace('/', '')
        model_id = f"{sanitized_symbol}_{timeframe}_unified"
        
        model_folder_path = os.path.join(model_dir, model_id)
        metadata_path = os.path.join(model_folder_path, "metadata.json")

        if os.path.exists(metadata_path):
            model_age = time.time() - os.path.getmtime(metadata_path)
            if model_age < timedelta(days=7).total_seconds():
                print(f"  -> [Genesis] ({symbol}) ✅ Specialist model is up-to-date. Skipping.")
                return

        # --- Data Loading ---
        data_path = os.path.join('data', f"{sanitized_symbol}_{timeframe}_raw.csv")
        if os.path.exists(data_path):
            df_full = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
        else:
            print(f"  -> [Genesis] No local data found for {symbol} at {data_path}. Fetching from network...")
            df_full = asyncio.run(get_market_data(symbol, timeframe, limit=2160))
            if df_full.empty:
                print(f"  -> [Genesis] Skipping {symbol}: No data found.")
                return
            df_full.to_csv(data_path)

        # --- Feature Engineering & Labeling ---
        df_features = add_all_features(df_full.copy())
        atr_col = next((c for c in df_features.columns if c.startswith('ATRr')), 'ATRr_14')
        df_features['label'] = get_specialist_labels(df_features, atr_col, module_type='reversion')
        
        # Sanitize before dropping NaNs
        df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_features.dropna(subset=['label'], inplace=True)
        if df_features.empty:
            print(f"  -> [Genesis] Skipping {symbol}: Not enough data after processing.")
            return

        # --- Model Training ---
        print(f"[Genesis] ({symbol}) Training {model_type} model...")
        model_map = {"lightgbm": LGBMWrapper, "xgboost": XGBWrapper}
        model_class = model_map.get(model_type.lower())
        if not model_class:
            print(f"  -> [Genesis] ERROR: Unknown model type '{model_type}'. Skipping.")
            return
            
        model = model_class(n_estimators=500, learning_rate=0.05) # Using some default params
        model.fit(df_features)

        # --- Model Saving ---
        model_folder_path = os.path.join(model_dir, model_id)
        os.makedirs(model_folder_path, exist_ok=True)

        model_file_path = os.path.join(model_folder_path, "model.joblib")
        joblib.dump(model.model, model_file_path)

        metadata = {
            "model_id": model_id,
            "asset_symbol": symbol, # Keep original symbol in metadata
            "status": "active",
            "registration_time": datetime.utcnow().isoformat(),
            "validation_metrics": {}
        }
        metadata_path = os.path.join(model_folder_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"  -> [Genesis] ✅ Saved initial specialist model to {model_folder_path}")

    except BaseException as e:
        print(f"💥 ERROR processing specialist model for {symbol}: {e}")

if __name__ == "__main__":
    import multiprocessing as mp

    # Force spawn context for stability, especially with CUDA/LightGBM
    try:
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
            print("[Multiprocessing] Start method forced to 'spawn'.")
    except RuntimeError:
        print("[Multiprocessing] Warning: Start method already set.")
        pass # Ignore if already set

    parser = argparse.ArgumentParser(description='Genesis V3 - Parallel Specialist Model Trainer')
    parser.add_argument('--timeframe', type=str, default='15m', help='The timeframe to use (e.g., 15m)')
    parser.add_argument('--model_dir', type=str, default='models', help='The directory to save models to')
    parser.add_argument('--model_type', type=str, default='lightgbm', help='The model type to train (lightgbm or xgboost)')
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    tasks = [(asset, args.timeframe, args.model_dir, args.model_type) for asset in ASSET_UNIVERSE]
    
    num_processes = min(cpu_count(), len(tasks))
    print(f"\n[Genesis] --- Starting Parallel Specialist Model Training ({num_processes} processes) ---")

    with Pool(processes=num_processes) as pool:
        pool.map(train_initial_model, tasks)

    print("\n[Genesis] --- Specialist Model Training Complete ---")
