import os
from dotenv import load_dotenv
import logging
import torch

# Load environment variables from .env file
load_dotenv()

# =================================================================================
# System Configuration
# =================================================================================

# --- Hardware Acceleration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Exchange Configuration ---
ACTIVE_EXCHANGE = "bybit"
BYBIT_USE_TESTNET = os.getenv('BYBIT_USE_TESTNET', 'false').lower() == 'true'

# --- INTEGRATED: DEPLOYMENT STRATEGY from backup ---
# This is the authoritative definition of which models to run.
DEPLOYMENT_STRATEGY = {
    'BTC/USDT:USDT': 4,  # 4 independent BTC models
    'ETH/USDT:USDT': 2,  # 2 independent ETH models
    'SOL/USDT:USDT': 2,  # 2 independent SOL models
}

# The ASSET_UNIVERSE is derived from the deployment strategy, preserving original logic.
ASSET_UNIVERSE = list(DEPLOYMENT_STRATEGY.keys())

# --- Exchange Configuration (loaded from .env file) ---
# PRESERVED: This logic is still needed by the crucible engine
# It needs a full list of potential assets for API key loading.
FULL_ASSET_UNIVERSE = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT", 
    "DOGE/USDT:USDT", "BNB/USDT:USDT", "ASTER/USDT:USDT", "ENA/USDT:USDT", 
    "SUI/USDT:USDT", "ADA/USDT:USDT"
]
BYBIT_SUBACCOUNT_KEYS = {
    asset: {
        'apiKey': os.getenv(f'BYBIT_API_KEY_{asset.split(":")[0].replace("/", "")}'),
        'secret': os.getenv(f'BYBIT_API_SECRET_{asset.split(":")[0].replace("/", "")}'),
    } for asset in FULL_ASSET_UNIVERSE
}

# --- Trading Manager ---
# PRESERVED: Important for live trading simulation
AGENT_VIRTUAL_BALANCE = 200

# --- INTEGRATED: Timeframe Configuration from updatev17.txt ---
TIMEFRAMES = {
    'strategic': '1h',
    'tactical': '15m',
    'microstructure': '1m'
}

# --- Directories ---
# PRESERVED: Core directory paths
DATA_DIR = 'data'
MODEL_DIR = 'models'

# --- Forge, Model, and Risk Parameters ---
# PRESERVED: These are orthogonal to the blueprint and still required
FORGE_CYCLE_COOLDOWN_MINUTES = 60
HEDGE_ENSEMBLE_LEARNING_RATE = 0.5
HMM_N_COMPONENTS = 4
LIVE_ATR_SL_MULTIPLIER = 2.0
LIVE_ATR_TP_MULTIPLIER = 4.0
ACTION_SHIELD_LIMITS = {
    "velocity_cap": 0.05,
    "volatility_factor": 0.5
}
DAILY_LOSS_LIMIT = 0.10
PEAK_DRAWDOWN_HALT = 0.30
EVENT_GUARDRAILS_ACTIVE = True
EVENT_BLACKOUT_WINDOW = 60
EVENT_FORCE_CLOSE_WINDOW = 30
LOG_LEVEL = logging.INFO
NUM_PARALLEL_WORKERS = 4 # Default number of parallel workers

# --- Feature Engineering Flags ---
ENABLE_FEATURE_SYNTHESIZER = True
ENABLE_CAUSAL_DISCOVERY = True

# --- INTEGRATED: ACN & Velocity Configuration from updatev17.txt ---
# This replaces the old ACN_CONFIG and transaction cost variables
ACN_CONFIG = {
    # Data Windows
    'training_window_mft': 5000,

    # Velocity Objective (Time-to-Target - TTT)
    'bt_initial_capital': 1000.0,
    'bt_velocity_target_pct': 2.0,

    # Realistic MFT Costs
    'bt_fees_pct': 0.055,
    'bt_spread_bps': 1.5,
    'bt_slippage_factor': 0.003,

    # GP 2.0 (StrategySynthesizer) Parameters
    'gp_population_size': 2001,
    'gp_generations': 101,
    'gp_parsimony_coeff': 0.01,
    'gp_tournament_size': 5,

    # Symbiotic Distillation (SDE) Weight
    'gp_symbiotic_mimicry_weight': 0.3,

    # --- Triple Barrier Labeling ---
    'PT_SL_ATR_MULTIPLE': [2.0, 2.0],
    'TARGET_ATR_MULTIPLE': 2.0,
    'MIN_RETURN': 0.001,
    'LOOKAHEAD_BARS': 20,
    'VOLATILITY_LOOKBACK': 100,
}

# --- Genetic Algorithm (GA) Parameters (for Model Blueprint GA) ---
GA_MIN_HORIZON = 10 # Example value, adjust as needed
GA_MAX_HORIZON = 60 # Example value, adjust as needed
GA_POPULATION_SIZE = ACN_CONFIG['gp_population_size']
GA_GENERATIONS = ACN_CONFIG['gp_generations']
USE_SURROGATE_ASSISTANCE = False # Default to False, enable if surrogate models are stable

print(f"[Config] Exchange Mode: {'TESTNET' if BYBIT_USE_TESTNET else 'PRODUCTION'}")
print(f"[Config] Deployment Strategy: {sum(DEPLOYMENT_STRATEGY.values())} models across {len(DEPLOYMENT_STRATEGY)} assets")
print(f"[Config] Active Assets: {ASSET_UNIVERSE}")
