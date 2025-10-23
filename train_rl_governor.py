# train_rl_governor.py
import os
# --- CRITICAL: Prevent PyTorch DLL issues on Windows ---
# This block MUST come before any imports that might load conflicting DLLs (like numpy, pandas, or torch)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
# ---------------------------------------------------

# Import the problematic library immediately after setting the environment variables
from stable_baselines3 import PPO

# Now import the rest of the libraries
import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib

# --- This block allows the script to be run from the root directory ---
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# -------------------------------------------------------------------

from forge.execution.rl_governor_env import GovernorEnv
from governor_market_simulator import GovernorMarketSimulator
from data_processing_v2 import add_all_features
from config import ASSET_UNIVERSE

def generate_base_signals(df: pd.DataFrame) -> pd.Series:
    """
    Generates a simple baseline trading strategy signal.
    1 = Buy, 2 = Exit, 0 = Hold
    """
    df['fast_ma'] = ta.sma(df['close'], length=10)
    df['slow_ma'] = ta.sma(df['close'], length=30)
    
    signals = pd.Series(np.zeros(len(df)), index=df.index)
    signals[df['fast_ma'] > df['slow_ma']] = 1 # Entry signal
    signals[df['fast_ma'] < df['slow_ma']] = 2 # Exit signal
    
    return signals

def main():
    """Trains the RL Governor for each asset in the universe."""
    print("[RL Governor Trainer] Starting RL Governor Training...")

    for symbol in ASSET_UNIVERSE:
        print(f"--- Training Governor for {symbol} ---")
        
        timeframe = '1h' # Use a longer timeframe for strategic decisions
        base_symbol = symbol.split(':')[0]
        sanitized_symbol = base_symbol.replace('/', '')
        data_path = os.path.join('data', f"{sanitized_symbol}_{timeframe}_raw.csv")

        if not os.path.exists(data_path):
            print(f"[RL Governor Trainer] WARNING: Could not find data for {symbol} at {data_path}. Skipping.")
            continue

        df_full = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
        df_features = add_all_features(df_full.copy(), symbol)
        
        # --- Add features needed for the Governor's observation space ---
        df_features['volatility'] = df_features.ta.atr(length=14)
        df_features['sharpe_ratio'] = (df_features['close'].pct_change().rolling(window=50).mean() / df_features['close'].pct_change().rolling(window=50).std()) * np.sqrt(252)
        df_features['confidence'] = 0.75 # Placeholder
        df_features['kelly_fraction'] = 0.5 # Placeholder
        df_features.dropna(inplace=True)
        
        base_signals = generate_base_signals(df_features.copy())

        if df_features.empty:
            print(f"[RL Governor Trainer] CRITICAL: Dataframe for {symbol} is empty after processing. Skipping.")
            continue
            
        print(f"[RL Governor Trainer] Data prepared for {symbol}. Shape: {df_features.shape}")

        simulator = GovernorMarketSimulator(df_features, base_signals)
        env = GovernorEnv(simulator)

        print("[RL Governor Trainer] Initializing PPO agent...")
        model = PPO("MultiInputPolicy", env, verbose=1)
        print("[RL Governor Trainer] Starting agent training...")
        model.learn(total_timesteps=20000)
        print("[RL Governor Trainer] Agent training complete.")

        # Save the model using the library's recommended method
        save_path = os.path.join("models", f"{sanitized_symbol}_RLG.zip")
        model.save(save_path)
        print(f"[SUCCESS] RL Governor for {symbol} saved to {save_path}")

if __name__ == "__main__":
    main()
