# train_rl_executor.py
import pandas as pd
import numpy as np
import os
import asyncio
from stable_baselines3 import PPO
import pandas_ta as ta

# --- This block allows the script to be run from the root directory ---
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# -------------------------------------------------------------------

from forge.execution.rl_executor import ExecutionEnv
from market_simulator import SimpleMarketSimulator
from data_processing_v2 import add_all_features, get_market_data
from config import ASSET_UNIVERSE

def simulate_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates realistic microstructure features from OHLCV data.
    This is a robust way to train the IEL agent without live order book data.
    """
    # Simulate Order Flow Imbalance (OFI) based on volume and price change
    price_change = df['close'].diff()
    df['ofi'] = (price_change * df['volume']).fillna(0)
    df['ofi'] = (df['ofi'] - df['ofi'].rolling(window=50).mean()) / df['ofi'].rolling(window=50).std() # Normalize

    # Simulate Book Pressure based on high/low distance from close
    df['book_pressure'] = ((df['high'] - df['close']) - (df['close'] - df['low'])) / (df['high'] - df['low'])
    df['book_pressure'].fillna(0, inplace=True)

    # Simulate Liquidity Density (as a function of volume)
    df['liquidity_density'] = df['volume'] / (df['high'] - df['low'])
    df['liquidity_density'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['liquidity_density'].fillna(method='ffill', inplace=True)

    # Use ATR for volatility, which is a standard feature
    df['volatility'] = df.ta.atr(length=14)

    # Simulate confidence (placeholder, can be a more complex signal)
    df['confidence'] = 0.75 # Assume a baseline confidence for training

    return df

def main():
    """Trains the RL execution agent on real, feature-rich data from the entire asset universe."""
    print("[IEL Trainer] Starting IEL Agent Training...")

    all_dfs = []

    for symbol in ASSET_UNIVERSE:
        print(f"--- Processing data for {symbol} ---")
        
        timeframe = '15m'
        base_symbol = symbol.split(':')[0]
        sanitized_symbol = base_symbol.replace('/', '')
        data_path = os.path.join('data', f"{sanitized_symbol}_{timeframe}_raw.csv")

        if os.path.exists(data_path):
            df_full = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
        else:
            print(f"[IEL Trainer] WARNING: Could not find local data for {symbol} at {data_path}. Fetching from network...")
            df_full = asyncio.run(get_market_data(symbol, timeframe, limit=5000))
            if df_full.empty:
                print(f"[IEL Trainer] WARNING: Could not fetch data for {symbol}. Skipping.")
                continue
            df_full.to_csv(data_path)
        
        df_features = add_all_features(df_full.copy(), symbol)
        
        # --- CRITICAL FIX: Add the missing microstructure features ---
        df_features = simulate_microstructure_features(df_features)
        # ---------------------------------------------------------

        all_dfs.append(df_features)
        print(f"[IEL Trainer] Features added for {symbol}. Shape: {df_features.shape}")

    if not all_dfs:
        print("[IEL Trainer] CRITICAL: No data could be processed for any asset. Halting.")
        return

    print("\n[IEL Trainer] Combining data from all assets...")
    combined_df = pd.concat(all_dfs)
    print(f"[IEL Trainer] Combined data shape: {combined_df.shape}")

    print("[IEL Trainer] Sanitizing combined data...")
    combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # This check is now for safety, but the simulation should prevent it from triggering
    required_cols = ['ofi', 'book_pressure', 'liquidity_density', 'volatility', 'confidence', 'close']
    for col in required_cols:
        if col not in combined_df.columns:
            print(f"[IEL Trainer] FATAL: Column '{col}' is still missing. Halting.")
            return
            
    combined_df.ffill(inplace=True)
    combined_df.bfill(inplace=True)
    combined_df.dropna(subset=['close'], inplace=True)

    if combined_df.empty:
        print("[IEL Trainer] CRITICAL: Combined dataframe is empty after processing. Halting.")
        return
        
    print(f"[IEL Trainer] Data sanitized. Final shape: {combined_df.shape}")

    print("[IEL Trainer] Initializing Market Simulator and Execution Environment...")
    simulator = SimpleMarketSimulator(combined_df)
    env = ExecutionEnv(simulator)
    print("[IEL Trainer] Environment initialized.")

    print("[IEL Trainer] Initializing PPO agent...")
    model = PPO("MultiInputPolicy", env, verbose=1)
    print("[IEL Trainer] Starting agent training (this may take a while)...")
    model.learn(total_timesteps=50000)
    print("[IEL Trainer] Agent training complete.")

    model.save("IEL_agent")
    print("\n[SUCCESS] RL Executor agent saved to IEL_agent.zip")

if __name__ == "__main__":
    main()