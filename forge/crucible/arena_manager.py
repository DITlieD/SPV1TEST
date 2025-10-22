import asyncio
import pandas as pd
import numpy as np
import logging
import functools # Import functools
from forge.core.agent import V3Agent

class CrucibleAgent:
    """
    Represents a single competitor in the Arena. It encapsulates a model,
    its virtual capital, and its performance metrics.
    """
    def __init__(self, agent_id: str, v3_agent: V3Agent, dna=None):
        self.agent_id = agent_id
        self.agent = v3_agent
        self.initial_capital = v3_agent.virtual_balance
        self.performance_history = []
        self.is_active = True
        self.dna = dna # Store the DNA

    def get_status(self):
        agent_status = self.agent.get_status()
        agent_status.update({
            "agent_id": self.agent_id,
            "model_id": self.agent.current_model_id,
            "is_active": self.is_active
        })
        return agent_status

class ArenaManager:
    """
    Manages the simultaneous execution and lifecycle of multiple CrucibleAgents.
    """
    def __init__(self, app_config, hedge_manager):
        self.config = app_config
        self.hedge_manager = hedge_manager
        self.agents = {} # {symbol: [CrucibleAgent, ...]}
        self._agents_lock = asyncio.Lock()  # FIX: Race condition protection
        self.asset_specific_models = {} # {symbol: {'regime_model': DynamicRegimeModel, ...}}
        self.stop_event = asyncio.Event()
        self.decisions_logger = self._setup_decisions_logger()

    def _setup_decisions_logger(self):
        """Sets up a dedicated logger for trading decisions."""
        logger = logging.getLogger("DecisionsLogger")
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()
        handler = logging.FileHandler("decisions_log.txt", mode='a')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    async def add_agent(self, symbol: str, agent: CrucibleAgent):
        """Adds a new agent to the arena."""
        async with self._agents_lock:  # FIX: Thread-safe access
            if symbol not in self.agents:
                self.agents[symbol] = []
            self.agents[symbol].append(agent)
        self.decisions_logger.info(f"AGENT ADDED: {agent.agent_id} to {symbol} arena.")

    def get_agents_for_symbol(self, symbol: str):
        """Returns the list of agents for a given symbol."""
        return self.agents.get(symbol, [])

    def get_agent_by_id(self, symbol: str, agent_id: str):
        """Safely retrieves an agent object by its ID."""
        agents = self.agents.get(symbol, [])
        for agent in agents:
            if agent.agent_id == agent_id:
                return agent
        return None

    async def remove_agent(self, symbol: str, agent_id: str):
        """Removes a specific agent from the arena."""
        async with self._agents_lock:  # FIX: Thread-safe access
            if symbol in self.agents:
                initial_count = len(self.agents[symbol])
                self.agents[symbol] = [agent for agent in self.agents[symbol] if agent.agent_id != agent_id]
                if len(self.agents[symbol]) < initial_count:
                    self.decisions_logger.info(f"AGENT REMOVED: {agent_id} from {symbol} arena.")

    async def replace_agent(self, symbol: str, old_agent_id: str, new_agent: 'CrucibleAgent'):
        """Replaces an existing agent with a new one (hot-swap) and updates the Hedge Manager."""
        # Notify the Hedge Manager of the change
        if self.hedge_manager:
            # It's assumed the agent_id is the same as the model_id for hot-swapping purposes
            self.hedge_manager.remove_model(old_agent_id)
            self.hedge_manager.add_model(new_agent.agent_id)

        # Remove the old agent
        await self.remove_agent(symbol, old_agent_id)
        # Add the new agent
        await self.add_agent(symbol, new_agent)
        self.decisions_logger.info(f"AGENT HOT-SWAP: {old_agent_id} replaced by {new_agent.agent_id} in {symbol} arena.")


    async def process_data(self, symbol: str, data: pd.DataFrame, sentiment_score: float, strategic_bias: str, forecasted_volatility: float, risk_multiplier: float = 1.0, mce_skew: float = 0.0, mce_pressure: float = 0.0, influence_incoming: float = 0.0, influence_outgoing: float = 0.0, predicted_entropy_delta: float = 0.0, regime_change_detected: bool = False):
        """
        Receives new feature data and passes it to all relevant agents for decisions.
        """
        # --- 1. Regime Classification ---
        current_regime = -1 # Default to noise/unknown
        regime_model = self.asset_specific_models.get(symbol, {}).get('regime_model')
        
        if regime_model and not data.empty:
            # CRITICAL FIX: Offload CPU-bound HMM prediction to a thread pool
            loop = asyncio.get_running_loop()
            try:
                features_for_regime = data[regime_model.feature_columns]
                # PRE-PREDICTION CHECK
                if features_for_regime.isnull().values.any() or np.isinf(features_for_regime.values).any():
                    self.decisions_logger.warning(f"Regime model input for {symbol} contains NaN or Inf. Skipping regime prediction.")
                else:
                    # Use functools.partial to pass arguments to the function run in the executor
                    predict_func = functools.partial(regime_model.predict_next_regime_proba, features_for_regime)
                    # Run in the default thread pool executor (None)
                    next_regime_probas = await loop.run_in_executor(None, predict_func)
                    
                    if len(next_regime_probas) > 0:
                        current_regime = np.argmax(next_regime_probas[-1]) # Get the most likely next regime
            except Exception as e:
                print(f"[{symbol}] Error during regime prediction offloading: {e}")

        # FIX: Thread-safe access to self.agents with lock
        async with self._agents_lock:
            if symbol in self.agents:
                active_agents = [agent for agent in self.agents[symbol] if agent.is_active]
            else:
                active_agents = []

        if active_agents:
            # FIX: Execute all agent decisions CONCURRENTLY for performance
            decision_tasks = [
                agent.agent.decide_and_execute(data, sentiment_score, strategic_bias, current_regime, risk_multiplier, mce_skew, mce_pressure, influence_incoming, influence_outgoing, predicted_entropy_delta, regime_change_detected)
                for agent in active_agents
            ]
            actions = await asyncio.gather(*decision_tasks, return_exceptions=True)

            bankrupt_agents = []  # Track agents that went bankrupt

            # Process results
            for agent, action in zip(active_agents, actions):
                if isinstance(action, Exception):
                    self.decisions_logger.error(f"ERROR: {agent.agent_id} decision failed: {action}")
                    continue

                if action and action.get('action'):
                    # Check for bankruptcy
                    if action['action'] == 'BANKRUPT':
                        self.decisions_logger.error(f"💀 BANKRUPT: {agent.agent_id} | {action['symbol']} | Reason: {action.get('reason', 'N/A')}")
                        bankrupt_agents.append(agent.agent_id)
                        agent.is_active = False  # Deactivate immediately
                    elif action['action'] != 'HOLD':
                        self.decisions_logger.info(f"DECISION: {agent.agent_id} | {action['action']} {action['symbol']} @ {action.get('price', 'N/A')} | Reason: {action.get('reason', 'N/A')}")

            # Remove bankrupt agents from arena
            for agent_id in bankrupt_agents:
                await self.remove_agent(symbol, agent_id)
                self.decisions_logger.warning(f"🗑️ REMOVED: {agent_id} from arena due to bankruptcy")

    def get_all_statuses(self):
        """Returns the status of all agents in the arena."""
        all_statuses = {}
        for symbol, agent_list in self.agents.items():
            all_statuses[symbol] = [agent.get_status() for agent in agent_list]
        return all_statuses

    def broadcast_hedge_weights(self, losses: dict):
        """Broadcasts the updated losses to all agents in the arena."""
        if not losses:
            return
        for agent_list in self.agents.values():
            for agent in agent_list:
                if agent.is_active:
                    agent.agent.update_hedge_weights(losses)

    async def rescan_and_update_worker(self, symbol: str):
        """Finds all agents for a symbol and triggers their model reload."""
        self.decisions_logger.info(f"ARENA: Received rescan signal for {symbol}.")
        agents_for_symbol = self.get_agents_for_symbol(symbol)
        if not agents_for_symbol:
            self.decisions_logger.warning(f"ARENA: No agents found for {symbol} to rescan.")
            return

        tasks = [agent.agent._reload_models() for agent in agents_for_symbol if agent.is_active]
        if tasks:
            await asyncio.gather(*tasks)
            self.decisions_logger.info(f"ARENA: Model reload complete for all active agents of {symbol}.")