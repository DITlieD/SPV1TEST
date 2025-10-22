import os
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# =================================================================================
# System Configuration
# =================================================================================

import torch

# --- Hardware Acceleration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # Can be 'cuda' or 'cpu'

# --- Exchange Configuration ---
ACTIVE_EXCHANGE = "bybit"
BYBIT_USE_TESTNET = os.getenv('BYBIT_USE_TESTNET', 'false').lower() == 'true'

# --- Asset Universe ---
# Full asset universe (all assets we want to track)
FULL_ASSET_UNIVERSE = [
    'BTC/USDT:USDT', 'XRP/USDT:USDT', 'MNT/USDT:USDT', 'ETH/USDT:USDT', 'SUI/USDT:USDT',
    'BNB/USDT:USDT', 'EPIC/USDT:USDT', 'ARC/USDT:USDT', 'SOL/USDT:USDT', 'DOGE/USDT:USDT'
]

# Assets available ONLY on testnet (not on production)
TESTNET_ONLY_ASSETS = ['EPIC/USDT:USDT', 'ARC/USDT:USDT']

# =================================================================================
# DEPLOYMENT STRATEGY - Multi-Model Per Asset (Update V5)
# =================================================================================
# Focus computational resources on core assets with multiple distinct models
DEPLOYMENT_STRATEGY = {
    'BTC/USDT:USDT': 4,  # 4 independent BTC models
    'ETH/USDT:USDT': 2,  # 2 independent ETH models
    'SOL/USDT:USDT': 2,  # 2 independent SOL models
}
# Total: 8 concurrent bots focused on high-liquidity assets

# The ASSET_UNIVERSE is derived from the deployment strategy
ASSET_UNIVERSE = list(DEPLOYMENT_STRATEGY.keys())

print(f"[Config] Exchange Mode: {'TESTNET' if BYBIT_USE_TESTNET else 'PRODUCTION'}")
print(f"[Config] Deployment Strategy: {sum(DEPLOYMENT_STRATEGY.values())} models across {len(DEPLOYMENT_STRATEGY)} assets")
print(f"[Config] Active Assets: {ASSET_UNIVERSE}")

# --- Exchange Configuration (loaded from .env file) ---
BYBIT_SUBACCOUNT_KEYS = {
    asset: {
        'apiKey': os.getenv(f'BYBIT_API_KEY_{asset.split(":")[0].replace("/", "")}'),
        'secret': os.getenv(f'BYBIT_API_SECRET_{asset.split(":")[0].replace("/", "")}'),
    } for asset in FULL_ASSET_UNIVERSE  # Generate keys for all assets
}

# --- Trading Manager ---
AGENT_VIRTUAL_BALANCE = 200

# MTFA (Multi-Timeframe Analysis) Configuration
# Strategic -> Tactical -> Microstructure hierarchy for noise reduction
TIMEFRAMES = {
    'strategic': '1h',       # Major trend context and regime definition
    'tactical': '15m',       # Intermediate structure and setup confirmation
    'microstructure': '1m'   # PRIMARY execution timeframe (ultra-aggressive MFT)
}

# --- Transaction Costs ---
FEE_PERCENT = 0.001  # 0.1% fee per trade
SLIPPAGE_PERCENT = 0.0005 # 0.05% slippage per trade

# --- Directories ---
DATA_DIR = 'data'
MODEL_DIR = 'models'

# --- Forge Configuration ---
FORGE_CYCLE_COOLDOWN_MINUTES = 60
HEDGE_ENSEMBLE_LEARNING_RATE = 0.5

# --- Model Parameters ---
HMM_N_COMPONENTS = 4
LIVE_ATR_SL_MULTIPLIER = 2.0
LIVE_ATR_TP_MULTIPLIER = 4.0

# --- Risk Management ---
ACTION_SHIELD_LIMITS = {
    "velocity_cap": 0.05,
    "volatility_factor": 0.5
}
DAILY_LOSS_LIMIT = 0.10
PEAK_DRAWDOWN_HALT = 0.30

# --- Event Guardrails ---
EVENT_GUARDRAILS_ACTIVE = True
EVENT_BLACKOUT_WINDOW = 60
EVENT_FORCE_CLOSE_WINDOW = 30

LOG_LEVEL = logging.INFO

# --- ACN Configuration ---
ACN_CONFIG = {
    'training_window': 10000, # Default training window size (bars)
    # Influence Mapper
    'im_entropy_window': 60,
}
