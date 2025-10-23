# forge/core/agent.py
import os
import joblib
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime
import json
import logging
import dill

# Local imports
import config
from utils import EconomicCalendar
from rl_governor import RLGovernor
from risk_management import CircuitBreaker, ActionShield
from models_v2 import LGBMWrapper, XGBWrapper, RiverOnlineModel
from forge.modeling.volatility_engine import VolatilityEngine
from forge.armory.model_registry import ModelRegistry
from validation_gauntlet import EvolvedStrategyWrapper
from forge.core.agent_state import state_manager

def calculate_fractional_kelly(win_prob, risk_reward_ratio, fraction=0.5):
    if risk_reward_ratio <= 0: return 0.0
    full_kelly = win_prob - ((1 - win_prob) / risk_reward_ratio)
    return max(0.0, full_kelly * fraction)

class V3Agent:
    """
    The core autonomous trading agent.
    """
    def __init__(self, symbol: str, exchange, app_config, agent_id: str = None, iel_agent=None, cerebellum_link=None):
        self.symbol = symbol
        self.exchange = exchange
        self.config = app_config
        self.sanitized_symbol = symbol.split(':')[0].replace('/', '')
        self.agent_id = agent_id or f"{self.sanitized_symbol}_agent"
        self.iel_agent = iel_agent
        self.cerebellum_link = cerebellum_link

        # --- Logger Setup ---
        self.logger = logging.getLogger(f"V3Agent_{self.sanitized_symbol}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = True
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(f'%(asctime)s - [V3Agent_{self.sanitized_symbol}] - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

        # --- State Persistence ---
        initial_state = state_manager.get_state(self.agent_id)
        if initial_state:
            self.logger.info(f"Restoring agent {self.agent_id} from saved state.")
            self.in_position = initial_state.get('in_position', False)
            self.current_trade = initial_state.get('current_trade', {})
            self.virtual_balance = initial_state.get('virtual_balance', getattr(self.config, 'AGENT_VIRTUAL_BALANCE', 1000))
            self.is_active = initial_state.get('is_active', True)
        else:
            self.logger.info(f"No saved state for {self.agent_id}. Initializing with default values.")
            self.in_position = False
            self.current_trade = {}
            self.virtual_balance = getattr(self.config, 'AGENT_VIRTUAL_BALANCE', 1000)
            self.is_active = True
            self._save_current_state()

        self.current_model_id = "None"
        
        # Initialize components
        self.model_registry = ModelRegistry(registry_path=self.config.MODEL_DIR)
        self.governor = self._load_rl_governor()
        self.hdbscan_model = None
        self.pending_trade_details = {}

        self.specialist_models = self._load_specialist_models()
        self._update_current_model_id() # <--- ADD THIS
        
        self.logger.info(f"V3Agent initialized for {self.symbol} with balance: ${self.virtual_balance:.2f}")

    def handle_fill_report(self, report_data):
        side = report_data.get("side")
        avg_price = float(report_data.get("avg_price"))
        executed_qty = float(report_data.get("executed_qty"))
        
        if side == "Buy":
            self.in_position = True
            self.current_trade = {
                'entry_price': avg_price,
                'amount': executed_qty,
                'entry_time': datetime.now().isoformat(),
                'capital_deployed': avg_price * executed_qty,
                'commission_paid': avg_price * executed_qty * 0.0007, # Assuming 0.07% commission
            }
            # --- ACTIVATE RISK MANAGEMENT ---
            if self.pending_trade_details:
                self.current_trade.update(self.pending_trade_details)
                self.logger.info(f"Activated SL/TP for current trade: SL={self.current_trade.get('stop_loss'):.2f}, TP={self.current_trade.get('take_profit'):.2f}")
                self.pending_trade_details = {} # Clear after use
            # --- END ---
            self._finalize_position(signal='ENTRY', amount=executed_qty, price=avg_price, atr=None, features=None)
        elif side == "Sell":
            self.in_position = False
            self._finalize_exit(exit_price=avg_price, reason="FILLED")
            self.current_trade = {}

        self.virtual_balance -= avg_price * executed_qty * (1 if side == "Buy" else -1)
        self._save_current_state()

    def _save_current_state(self):
        """Saves the agent's current state to the persistent store."""
        state = {
            "agent_id": self.agent_id,
            "symbol": self.symbol,
            "in_position": self.in_position,
            "current_trade": self.current_trade,
            "virtual_balance": self.virtual_balance,
            "is_active": self.is_active,
            "timestamp": datetime.now().isoformat()
        }
        state_manager.save_state(self.agent_id, state)
        self.logger.debug(f"Agent {self.agent_id} state saved.")

    def _update_current_model_id(self):
        # Use the agent's unique ID (derived from the model directory)
        # as the identifier displayed in the UI.
        if self.agent_id:
             self.current_model_id = self.agent_id
             self.logger.info(f"Set current_model_id to: {self.current_model_id}")
        # Optional: Add logic here if you need to select a *specific* model
        # from self.specialist_models based on some criteria.
        # For now, using self.agent_id aligns with how agents are created.

    def get_status(self):
        """Returns the current status of the agent."""
        # Determine current activity status
        status_message = "Idle"
        if self.in_position and self.current_trade:
            status_message = "In Position"
        elif self.is_active:
            status_message = "Searching for Trade"

        return {
            "symbol": self.symbol,
            "in_position": self.in_position,
            "current_trade": self.current_trade,
            "balance": self.virtual_balance,
            "model_id": self.current_model_id,
            "is_active": self.is_active,
            "status_message": status_message
        }

    def _load_hdbscan_model(self):
        try:
            model_path = os.path.join(self.config.MODEL_DIR, f"{self.sanitized_symbol}_HDBSCAN.joblib")
            if os.path.exists(model_path):
                self.logger.info(f"Loading HDBSCAN model from {model_path}")
                return joblib.load(model_path)
        except Exception as e:
            self.logger.error(f"Error loading HDBSCAN for {self.sanitized_symbol}: {e}")
        return None

    def _load_rl_governor(self):
        try:
            model_path = os.path.join(self.config.MODEL_DIR, f"{self.sanitized_symbol}_RLG.zip")
            if os.path.exists(model_path):
                self.logger.info(f"Loading Deep RL Governor from {model_path}")
                return RLGovernor.load(model_path, env=None)
        except Exception as e:
            self.logger.error(f"Error loading RL Governor for {self.sanitized_symbol}: {e}")
        return None

    def _load_specialist_models(self):
        models = {}
        models_dir = self.config.MODEL_DIR
        
        self.logger.info(f"Scanning for model directories in {models_dir} for symbol {self.symbol}")

        for item_name in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item_name)
            if os.path.isdir(item_path):
                metadata_path = os.path.join(item_path, 'metadata.json')
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        if metadata.get("asset_symbol") == self.symbol:
                            # This model belongs to this agent
                            model_id = metadata.get("model_id", item_name)
                            model_type = metadata.get("blueprint", {}).get("model_type")
                            
                            if model_type == "GP 2.0":
                                components_path = os.path.join(item_path, "strategy_components.dill")
                                if os.path.exists(components_path):
                                    from deap import base, creator, gp as deap_gp
                                    if not hasattr(creator, "FitnessMax"):
                                        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
                                    if not hasattr(creator, "Individual"):
                                        creator.create("Individual", deap_gp.PrimitiveTree, fitness=creator.FitnessMax)

                                    with open(components_path, 'rb') as f:
                                        gp_components = dill.load(f)

                                    from validation_gauntlet import EvolvedStrategyWrapper
                                    reconstructed_model = EvolvedStrategyWrapper(
                                        tree=gp_components['tree'],
                                        pset=gp_components['pset'],
                                        feature_names=gp_components['feature_names']
                                    )
                                    models[model_id] = reconstructed_model
                                    self.logger.info(f"Successfully loaded GP 2.0 model: {model_id}")
                                else:
                                    self.logger.error(f"GP 2.0 model {model_id}: strategy_components.dill not found at {components_path}")
                            
                            elif model_type == "MetaLearner_Reptile":
                                model_file_path = os.path.join(item_path, "model.pt")
                                if os.path.exists(model_file_path):
                                    import torch
                                    from forge.models.meta_learner import MetaLearner
                                    
                                    # Recreate the model structure
                                    # The feature names are needed to know the input dimension
                                    feature_names = metadata.get("features", [])
                                    if not feature_names:
                                        self.logger.error(f"Cannot load MetaLearner {model_id}: feature list not in metadata.")
                                        continue
                                    
                                    input_dim = len(feature_names)
                                    loaded_model = MetaLearner(input_dim=input_dim)
                                    loaded_model.load_state_dict(torch.load(model_file_path))
                                    loaded_model.to(self.config.DEVICE) # Move to GPU/CPU
                                    loaded_model.eval()

                                    # Wrap it for consistent API
                                    from forge.overlord.task_scheduler import MetaLearnerWrapper
                                    wrapped_model = MetaLearnerWrapper(loaded_model, features=feature_names, device=self.config.DEVICE)
                                    models[model_id] = wrapped_model
                                    self.logger.info(f"Successfully loaded MetaLearner model: {model_id}")
                            
                            else:
                                model_file_path = os.path.join(item_path, "model.joblib")
                                if os.path.exists(model_file_path):
                                    models[model_id] = joblib.load(model_file_path)
                                    self.logger.info(f"Successfully loaded joblib model: {model_id}")

                    except Exception as e:
                        self.logger.error(f"Error processing model directory {item_name}: {e}", exc_info=True)

        if not models:
            self.logger.warning(f"No specialist models found for {self.symbol}. Agent will be inactive.")
        else:
            self.logger.info(f"Loaded {len(models)} specialist models for {self.symbol}: {list(models.keys())}")

        return models

    async def decide_and_execute(self, features_df: pd.DataFrame, sentiment_score: float = 0.5, strategic_bias: str = 'NEUTRAL', regime_id: int = -1, risk_multiplier: float = 1.0, mce_skew: float = 0.0, mce_pressure: float = 0.0, influence_incoming: float = 0.0, influence_outgoing: float = 0.0, predicted_entropy_delta: float = 0.0, regime_change_detected: bool = False):
        self.logger.info(f"[DIAGNOSTIC] --- Entering decide_and_execute for {self.symbol} ---")
        self.logger.debug(f"TDA Regime Change Signal: {regime_change_detected}")

        if regime_change_detected:
            self.logger.warning("[DIAGNOSTIC] TDA Regime Change detected! Applying caution.")
            if self.in_position:
                self.logger.warning("[DIAGNOSTIC] Forcing exit due to TDA regime change.")
                await self._execute_trade(signal='EXIT', size_fraction=1.0, atr=0, latest_features=features_df.tail(1))
                return { 'action': 'SELL', 'symbol': self.symbol, 'price': features_df.tail(1)['close'].values[0], 'reason': 'TDA Regime Change Exit' }

        tda_risk_adj = 0.1 if regime_change_detected else 1.0

        if not self.is_active or features_df.empty:
            self.logger.info(f"[DIAGNOSTIC] Agent is inactive or dataframe is empty. Skipping.")
            return None
        if not self.specialist_models:
            self.logger.warning("[DIAGNOSTIC] No specialist models loaded. Cannot make a decision.")
            return None

        
        self.logger.info(f"[DIAGNOSTIC] {len(self.specialist_models)} specialist models are active.")
        latest_features = features_df.tail(1)
        current_price = latest_features['close'].values[0]
        atr = latest_features.get('atr', pd.Series([0])).values[0]

        # --- META-LEARNING FAST UPDATE (if applicable) ---
        for model_id, model in self.specialist_models.items():
            if hasattr(model, 'is_meta_learner') and model.is_meta_learner:
                self.logger.debug(f"Performing fast-update for MetaLearner model {model_id}...")
                try:
                    recent_data = features_df.tail(50)
                    model.fast_update(recent_data)
                except Exception as e:
                    self.logger.error(f"Error during MetaLearner fast-update for {model_id}: {e}")

        # --- SINGLE MODEL PREDICTION ---
        # Since the architecture is one agent per model, we don't need ensemble logic.
        model_id, model = next(iter(self.specialist_models.items()))

        try:
            if hasattr(model, 'feature_names'):
                required_features = model.feature_names
                missing = set(required_features) - set(latest_features.columns)
                if missing:
                    self.logger.warning(f"Model {model_id} requires missing features: {missing}. Skipping prediction.")
                    return None
                model_input = latest_features[required_features]
            else:
                model_input = latest_features

            if model_input.isnull().values.any() or np.isinf(model_input.values).any():
                self.logger.warning(f"Model {model_id} input contains NaN or Inf. Skipping prediction.")
                return None

            pred = model.predict(model_input)
            if isinstance(pred, np.ndarray): pred = pred[0] if len(pred) > 0 else 0
            elif isinstance(pred, pd.Series): pred = pred.values[0]
            
            signal = int(pred)
            confidence = 0.75 # Default confidence if model doesn't provide it

            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(model_input)
                    if isinstance(proba, np.ndarray) and proba.ndim == 2:
                        # Set confidence to the probability of the predicted class
                        confidence = proba[0][signal]
                except Exception:
                    self.logger.warning(f"Could not get probability from model {model_id}. Using default confidence.")
            
            self.logger.info(f"[DIAGNOSTIC] Model {model_id} produced signal={signal} with confidence={confidence:.2%}")

        except Exception as e:
            self.logger.error(f"Error getting prediction from model {model_id}: {e}")
            return None

        ensemble_signal = signal # Use the direct signal from the single model


        # --- Regime-based parameter adjustment ---
        if regime_id == 0: # Trending
            risk_reward = 3.5
        elif regime_id == 1: # Mean-reverting
            risk_reward = 2.5
        else: # Volatile/Noise
            risk_reward = 2.0

        # --- POSITION MANAGEMENT ---
        if self.in_position:
            self.logger.info("[DIAGNOSTIC] Agent is in position. Evaluating exit conditions.")
            # (Original exit logic is preserved here)
            # ...
            pass # This will fall through to the original logic below
        else:
            self.logger.info("[DIAGNOSTIC] Agent is not in position. Evaluating entry conditions.")
            if ensemble_signal == 1:  # ENTRY
                self.logger.info("[DIAGNOSTIC] ENTRY signal received. Evaluating position size.")
                win_prob = confidence
                kelly_fraction = calculate_fractional_kelly(win_prob, risk_reward, fraction=0.5)
                self.logger.info(f"[DIAGNOSTIC] Kelly Fraction calculated: {kelly_fraction:.4f}")
                size_fraction = min(kelly_fraction, 0.95) * risk_multiplier

                if size_fraction > 0.01:
                    self.logger.info(f"[DIAGNOSTIC] Position size ({size_fraction:.2%}) is valid. EXECUTING TRADE.")
                    # This will fall through to the original logic
                else:
                    self.logger.info(f"[DIAGNOSTIC] Position size ({size_fraction:.2%}) is too small. Holding position.")
            else:
                self.logger.info(f"[DIAGNOSTIC] Signal is not ENTRY ({ensemble_signal}). Holding position.")
        
        # --- ORIGINAL LOGIC (PRESERVED) ---
        if self.in_position:
            entry_price = self.current_trade.get('entry_price', current_price)
            stop_loss = self.current_trade.get('stop_loss', 0)
            take_profit = self.current_trade.get('take_profit', float('inf'))
            entry_time = self.current_trade.get('entry_time')
            
            if entry_time:
                minutes_in_position = (datetime.now() - datetime.fromisoformat(entry_time)).total_seconds() / 60
            else:
                minutes_in_position = 0
            
            atr_pct = (atr / current_price) * 100 if current_price > 0 else 1.0
            if atr_pct >= 2.5: adaptive_timeout = 3
            elif atr_pct >= 1.5: adaptive_timeout = 5
            elif atr_pct >= 1.0: adaptive_timeout = 7
            elif atr_pct >= 0.5: adaptive_timeout = 9
            else: adaptive_timeout = 10
            
            force_exit = False
            exit_reason = None
            if current_price <= stop_loss:
                force_exit = True
                exit_reason = f'Stop-loss hit @ ${stop_loss:.2f}'
            elif current_price >= take_profit:
                force_exit = True
                exit_reason = f'Take-profit hit @ ${take_profit:.2f}'
            elif minutes_in_position > adaptive_timeout:
                force_exit = True
                exit_reason = f'Adaptive timeout ({minutes_in_position:.0f}m / {adaptive_timeout}m limit, ATR: {atr_pct:.2f}%)'
            elif ensemble_signal == 2:
                force_exit = True
                exit_reason = f'Ensemble EXIT signal (confidence: {confidence:.2%})'
            
            if force_exit:
                await self._execute_trade(signal='EXIT', size_fraction=1.0, atr=atr, features=latest_features)
                return { 'action': 'SELL', 'symbol': self.symbol, 'price': current_price, 'reason': exit_reason }
            else:
                unrealized_pnl = (current_price - entry_price) * self.current_trade.get('amount', 0)
                unrealized_pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                self.logger.debug(f"Holding position: Unrealized P&L: ${unrealized_pnl:.2f} ({unrealized_pnl_pct:+.2f}%)")
                return None
        else:
            if ensemble_signal == 1:
                win_prob = confidence
                kelly_fraction = calculate_fractional_kelly(win_prob, risk_reward, fraction=0.5)
                size_fraction = min(kelly_fraction, 0.95) * risk_multiplier

                # --- RL Governor ---
                if self.governor:
                    try:
                        gov_obs = latest_features[self.governor.feature_names]
                        gov_action, _ = self.governor.predict(gov_obs, deterministic=True)
                        # Assuming governor action is a multiplier between 0.5 and 1.5
                        size_fraction *= gov_action[0]
                        self.logger.info(f"RL Governor adjusted size fraction by {gov_action[0]:.2f}")
                    except Exception as e:
                        self.logger.error(f"Error using RL Governor: {e}")

                # --- MCE Adjustment ---
                mce_confidence_adj = 1.0
                if mce_skew < -0.2: # Skew opposes LONG signal
                    mce_confidence_adj *= 0.5
                if mce_pressure > 0.8: # High pressure = risky entry
                    mce_confidence_adj *= 0.3
                size_fraction *= mce_confidence_adj
                self.logger.info(f"[DIAGNOSTIC] MCE Applied: Skew={mce_skew:.2f}, Pressure={mce_pressure:.2f}, AdjFactor={mce_confidence_adj:.2f}, Final Size={size_fraction:.2%}")

                # --- Influence Mapper Adjustment ---
                influence_risk_adj = max(0.1, 1.0 - abs(influence_incoming * 2.0)) # Scale influence to [0,1] first if needed
                size_fraction *= influence_risk_adj
                self.logger.info(f"[DIAGNOSTIC] Influence Risk Applied: Factor={influence_risk_adj:.3f}, Final Size={size_fraction:.2%}")

                # --- TDA Adjustment ---
                size_fraction *= tda_risk_adj
                self.logger.info(f"[DIAGNOSTIC] TDA Risk Applied: Factor={tda_risk_adj:.2f}, Final Size={size_fraction:.2%}")

                if size_fraction > 0.01:
                    # --- DYNAMIC RISK MANAGEMENT ---
                    stop_loss_distance = self.config.STOP_LOSS_ATR_MULTIPLE * atr
                    stop_loss_price = current_price - stop_loss_distance
                    
                    take_profit_price = current_price + (stop_loss_distance * risk_reward)
                    
                    self.pending_trade_details = {
                        'stop_loss': stop_loss_price,
                        'take_profit': take_profit_price
                    }
                    self.logger.info(f"Calculated SL: {stop_loss_price:.2f} | TP: {take_profit_price:.2f} (ATR: {atr:.4f}, R:R: {risk_reward})")
                    # --- END DYNAMIC RISK MANAGEMENT ---

                    await self._execute_trade(signal='ENTRY', size_fraction=size_fraction, atr=atr, features=latest_features)
                    return { 'action': 'BUY', 'symbol': self.symbol, 'price': current_price, 'size_fraction': size_fraction, 'reason': f'Ensemble ENTRY signal (confidence: {confidence:.2%}, Kelly: {kelly_fraction:.2%})' }
        return None

    async def _execute_trade(self, signal, size_fraction, atr, latest_features):
        """
        Executes a trade (ENTRY or EXIT) by sending a command to the Cerebellum Core.
        """
        if not self.cerebellum_link:
            self.logger.error("[Agent] Cerebellum link not available. Cannot execute trade.")
            return

        side = 'Buy' if signal == 'ENTRY' else 'Sell'
        
        try:
            ticker = await self.exchange.fetch_ticker(self.symbol)
            price = ticker['ask'] if side == 'Buy' else ticker['bid']
            if price is None or price == 0:
                price = latest_features['close'].values[0]
        except Exception as e:
            self.logger.error(f"Error fetching ticker for {self.symbol}: {e}. Falling back to close price.")
            price = latest_features['close'].values[0]

        quantity = self.virtual_balance * size_fraction / price

        # Use the IEL agent to decide the execution mode
        iel_mode = "Aggressive" # Default mode
        if self.iel_agent:
            iel_obs = {
                "ofi": np.array([latest_features['ofi'].values[0]], dtype=np.float32),
                "book_pressure": np.array([latest_features['book_pressure'].values[0]], dtype=np.float32),
                "liquidity_density": np.array([latest_features['liquidity_density'].values[0]], dtype=np.float32),
                "volatility": np.array([latest_features['volatility'].values[0]], dtype=np.float32),
                "confidence": np.array([latest_features['confidence'].values[0]], dtype=np.float32),
            }
            iel_action, _ = self.iel_agent.predict(iel_obs, deterministic=True)
            if iel_action == 0:
                iel_mode = "Aggressive"
            elif iel_action == 1:
                iel_mode = "Passive"
            else:
                iel_mode = "Adaptive" # Assuming 'Market' maps to 'Adaptive'

        self.cerebellum_link.execute_order(
            symbol=self.sanitized_symbol,
            side=side,
            quantity=quantity,
            strategy_id=self.agent_id,
            iel_mode=iel_mode
        )

        self.logger.info(f"[Agent] Sent {side} order for {quantity} of {self.symbol} to Cerebellum.")

    def _finalize_position(self, signal, amount, price, atr, features):
        """
        Logs entry position details to performance log.
        Records full state for later analysis and exit tracking.
        """
        if signal != 'ENTRY':
            return

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'agent_id': self.agent_id,
            'symbol': self.symbol,
            'model_id': self.current_model_id,
            'action': 'ENTRY',
            'price': float(price),
            'amount': float(amount),
            'capital_deployed': float(self.current_trade.get('capital_deployed', 0)),
            'commission': float(self.current_trade.get('commission_paid', 0)),
            'stop_loss': float(self.current_trade.get('stop_loss', 0)),
            'take_profit': float(self.current_trade.get('take_profit', 0)),
            'atr': float(atr) if atr is not None else 'N/A',
            'balance_before': float(self.virtual_balance + self.current_trade.get('capital_deployed', 0)),
            'balance_after': float(self.virtual_balance),
        }

        # Append to performance log
        try:
            with open('performance_log.jsonl', 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write entry to performance log: {e}")

    def _finalize_exit(self, exit_price, reason):
        """
        Logs exit position details and calculates full trade statistics.
        Records complete trade lifecycle for watchtower judgment.
        """
        if not self.current_trade:
            self.logger.warning("_finalize_exit called but no current_trade data available.")
            return

        # Calculate full trade metrics
        entry_price = self.current_trade.get('entry_price', 0)
        entry_time = self.current_trade.get('entry_time', datetime.now().isoformat())
        amount = self.current_trade.get('amount', 0)
        capital_deployed = self.current_trade.get('capital_deployed', 0)
        entry_commission = self.current_trade.get('commission_paid', 0)

        # Exit value and commission
        exit_value = amount * exit_price
        exit_commission = exit_value * 0.0007
        net_proceeds = exit_value - commission

        # PnL calculations
        gross_pnl = exit_value - capital_deployed
        net_pnl = net_proceeds - capital_deployed
        pnl_pct = (net_pnl / capital_deployed * 100) if capital_deployed > 0 else 0

        # Trade duration
        try:
            entry_dt = datetime.fromisoformat(entry_time)
            exit_dt = datetime.now()
            duration_seconds = (exit_dt - entry_dt).total_seconds()
            duration_hours = duration_seconds / 3600
        except Exception:
            duration_hours = 0

        # Risk-adjusted metrics
        price_change_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
        r_multiple = net_pnl / (capital_deployed * 0.02) if capital_deployed > 0 else 0  # Assuming 2% risk per trade

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'agent_id': self.agent_id,
            'symbol': self.symbol,
            'model_id': self.current_model_id,
            'action': 'EXIT',
            'entry_price': float(entry_price),
            'exit_price': float(exit_price),
            'entry_time': entry_time,
            'exit_time': datetime.now().isoformat(),
            'duration_hours': float(duration_hours),
            'amount': float(amount),
            'capital_deployed': float(capital_deployed),
            'entry_commission': float(entry_commission),
            'exit_commission': float(exit_commission),
            'total_commission': float(entry_commission + exit_commission),
            'gross_pnl': float(gross_pnl),
            'net_pnl': float(net_pnl),
            'pnl_pct': float(pnl_pct),
            'price_change_pct': float(price_change_pct),
            'r_multiple': float(r_multiple),
            'exit_reason': reason,
            'balance_after': float(self.virtual_balance),
            'stop_loss': float(self.current_trade.get('stop_loss', 0)),
            'take_profit': float(self.current_trade.get('take_profit', 0)),
        }

        # Append to performance log
        try:
            with open('performance_log.jsonl', 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            self.logger.info(f"Trade logged: {net_pnl:+.2f} USD ({pnl_pct:+.2f}%) over {duration_hours:.1f}h | R-Multiple: {r_multiple:.2f}")
        except Exception as e:
            self.logger.error(f"Failed to write exit to performance log: {e}")

        self.logger.debug(f"Hedge weight update received and processed: {self.hedge_weights}")
