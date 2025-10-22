"""
VELOCITY UPGRADE: Pit Crew Hot-Swap Architecture
=================================================
Continuous shadow evaluation with instant hot-swaps for superior models.

The Pit Crew Mechanism:
1. Forge runs continuously, producing models placed on Challenger Bench
2. CrucibleEngine evaluates Challengers in shadow mode (no real trades)
3. If Challenger shows better TTT than Champion, instant hot-swap occurs
4. System adapts in near real-time based on performance, not failure

Result: Near real-time adaptation to changing market conditions.
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import os


@dataclass
class ChallengerModel:
    """Represents a model waiting on the Challenger Bench."""
    model_id: str
    symbol: str
    generation: int
    added_time: datetime
    shadow_performance: List[float] = None  # TTT measurements in shadow mode
    shadow_trades: int = 0
    psr: float = 0.0
    drawdown: float = 0.0

    def __post_init__(self):
        if self.shadow_performance is None:
            self.shadow_performance = []

    def get_average_ttt(self) -> float:
        """Calculate average Time-to-Target from shadow trades."""
        if not self.shadow_performance:
            return float('inf')  # Worst possible TTT
        return np.mean(self.shadow_performance)

    def get_performance_score(self) -> float:
        """
        Calculate overall performance score.
        Lower is better (faster TTT = better).
        """
        if len(self.shadow_performance) < 3:
            # Not enough data yet
            return float('inf')

        avg_ttt = self.get_average_ttt()
        consistency = np.std(self.shadow_performance) if len(self.shadow_performance) > 1 else 0

        # Penalize inconsistency
        score = avg_ttt + (consistency * 0.5)
        return score


class PitCrew:
    """
    Manages the Challenger Bench and hot-swap decisions.

    Workflow:
    1. New models from Forge go to Challenger Bench
    2. Challengers are evaluated in shadow mode (virtual trades)
    3. When Challenger outperforms Champion, hot-swap occurs
    4. Old Champion may be demoted back to bench or discarded
    """

    def __init__(self, app_config, logger=None):
        self.config = app_config
        self.logger = logger or logging.getLogger(__name__)

        # Challenger Bench: {symbol: [ChallengerModel, ...]}
        self.challenger_bench: Dict[str, List[ChallengerModel]] = {}

        # Current Champions: {symbol: model_id}
        self.champions: Dict[str, str] = {}

        # Shadow mode state: Track virtual capital for challengers
        self.shadow_state: Dict[str, Dict[str, float]] = {}  # {symbol: {model_id: virtual_capital}}

        # Configuration
        self.min_shadow_trades = 5  # Minimum trades before considering swap
        self.performance_threshold = 0.9  # Challenger must be 10% better to swap
        self.max_bench_size = 5  # Maximum challengers per symbol

        self.logger.info("[PIT CREW] Initialized - Continuous shadow evaluation active")

    def add_challenger(self, symbol: str, model_id: str, generation: int):
        """
        Add a new model to the Challenger Bench.

        Args:
            symbol: Asset symbol
            model_id: Model identifier
            generation: Evolution generation number
        """
        if symbol not in self.challenger_bench:
            self.challenger_bench[symbol] = []

        challenger = ChallengerModel(
            model_id=model_id,
            symbol=symbol,
            generation=generation,
            added_time=datetime.now()
        )

        self.challenger_bench[symbol].append(challenger)

        # Initialize shadow state
        if symbol not in self.shadow_state:
            self.shadow_state[symbol] = {}
        self.shadow_state[symbol][model_id] = 200.0  # Starting virtual capital

        self.logger.info(f"[PIT CREW] Added challenger to bench: {model_id} for {symbol} (Gen {generation})")

        # Trim bench if too large
        self._trim_bench(symbol)

    def _trim_bench(self, symbol: str):
        """Remove worst performing challengers if bench is full."""
        if len(self.challenger_bench[symbol]) > self.max_bench_size:
            # Sort by performance (best first)
            self.challenger_bench[symbol].sort(key=lambda c: c.get_performance_score())

            # Keep only top performers
            discarded = self.challenger_bench[symbol][self.max_bench_size:]
            self.challenger_bench[symbol] = self.challenger_bench[symbol][:self.max_bench_size]

            # Clean up shadow state for discarded
            for challenger in discarded:
                if challenger.model_id in self.shadow_state.get(symbol, {}):
                    del self.shadow_state[symbol][challenger.model_id]
                self.logger.info(f"[PIT CREW] Discarded poor performer: {challenger.model_id}")

    def record_shadow_performance(self, symbol: str, model_id: str, time_to_target: int, final_capital: float, psr: float, drawdown: float):
        """
        Record performance from a shadow trade.

        Args:
            symbol: Asset symbol
            model_id: Model that made the trade
            time_to_target: Bars to reach profit target (-1 if not reached)
            final_capital: Ending virtual capital after trade
            psr: Probabilistic Sharpe Ratio
            drawdown: Max drawdown percentage
        """
        # Find the challenger
        challenger = self._find_challenger(symbol, model_id)
        if not challenger:
            return

        # Record TTT (use max bars if target not reached)
        ttt = time_to_target if time_to_target > 0 else 1000
        challenger.shadow_performance.append(ttt)
        challenger.shadow_trades += 1
        challenger.psr = psr
        challenger.drawdown = drawdown

        # Update shadow capital
        if symbol in self.shadow_state and model_id in self.shadow_state[symbol]:
            self.shadow_state[symbol][model_id] = final_capital

        self.logger.debug(f"[PIT CREW] Shadow trade recorded for {model_id}: TTT={ttt}, PSR={psr:.2f}, Drawdown={drawdown:.2%}, Capital={final_capital:.2f}")

    def _find_challenger(self, symbol: str, model_id: str) -> Optional[ChallengerModel]:
        """Find a challenger by model_id."""
        if symbol not in self.challenger_bench:
            return None

        for challenger in self.challenger_bench[symbol]:
            if challenger.model_id == model_id:
                return challenger
        return None

    def _get_agent_live_performance(self, agent_id: str) -> dict:
        """Reads performance_log.jsonl to get the live performance of an agent."""
        if not agent_id:
            return {"score": 100.0, "psr": 0.5, "drawdown": 0.1}
        try:
            # TODO: Implement parsing of performance_log.jsonl
            # For now, return default values
            return {"score": 100.0, "psr": 0.5, "drawdown": 0.1}
        except FileNotFoundError:
            return {"score": 100.0, "psr": 0.5, "drawdown": 0.1} # Default score if no log exists

    def check_for_hotswap(self, symbol: str, current_champion_id: str) -> Optional[str]:
        """
        Check if any challenger should replace the current champion.

        Returns:
            model_id of the new champion if swap should occur, None otherwise
        """
        if not current_champion_id:
            # No champion yet, promote best challenger if ready
            return self._promote_best_challenger(symbol)

        if symbol not in self.challenger_bench or not self.challenger_bench[symbol]:
            return None

        # Get current champion's performance
        champion_perf = self._get_agent_live_performance(current_champion_id)
        champion_score = champion_perf["score"]

        # Find best challenger
        best_challenger = None
        best_score = float('inf')

        for challenger in self.challenger_bench[symbol]:
            if challenger.shadow_trades < self.min_shadow_trades:
                continue  # Not enough data

            score = challenger.get_performance_score()
            if score < best_score:
                best_score = score
                best_challenger = challenger

        if not best_challenger:
            return None

        # Check if challenger is significantly better
        if (best_score < champion_score * self.performance_threshold and
            best_challenger.psr > 0.55 and
            best_challenger.drawdown < champion_perf["drawdown"] * 1.1): # Allow 10% more drawdown
            self.logger.info(
                f"[PIT CREW] 🔥 HOT-SWAP TRIGGERED for {symbol}! "
                f"Challenger {best_challenger.model_id} (Score: {best_score:.2f}, PSR: {best_challenger.psr:.2f}, DD: {best_challenger.drawdown:.2%}) "
                f"outperforms Champion {current_champion_id} (Score: {champion_score:.2f}, PSR: {champion_perf['psr']:.2f}, DD: {champion_perf['drawdown']:.2%})"
            )

            # Perform the swap
            old_champion = self.champions.get(symbol)
            self.champions[symbol] = best_challenger.model_id

            # Remove challenger from bench
            self.challenger_bench[symbol].remove(best_challenger)

            return best_challenger.model_id

        return None

    def _promote_best_challenger(self, symbol: str) -> Optional[str]:
        """Promote the best challenger to champion if ready."""
        if symbol not in self.challenger_bench or not self.challenger_bench[symbol]:
            return None

        # Find best challenger with enough trades
        best_challenger = None
        best_score = float('inf')

        for challenger in self.challenger_bench[symbol]:
            if challenger.shadow_trades < self.min_shadow_trades:
                continue

            score = challenger.get_performance_score()
            if score < best_score:
                best_score = score
                best_challenger = challenger

        if best_challenger:
            self.logger.info(
                f"[PIT CREW] Promoting initial champion for {symbol}: {best_challenger.model_id} "
                f"(Score: {best_score:.2f}, Trades: {best_challenger.shadow_trades})"
            )
            self.champions[symbol] = best_challenger.model_id
            self.challenger_bench[symbol].remove(best_challenger)
            return best_challenger.model_id

        return None

    def get_status(self, symbol: str = None) -> Dict:
        """Get current Pit Crew status."""
        if symbol:
            return {
                'symbol': symbol,
                'champion': self.champions.get(symbol, 'None'),
                'challengers': len(self.challenger_bench.get(symbol, [])),
                'challenger_details': [
                    {
                        'model_id': c.model_id,
                        'generation': c.generation,
                        'shadow_trades': c.shadow_trades,
                        'avg_ttt': c.get_average_ttt(),
                        'score': c.get_performance_score()
                    }
                    for c in self.challenger_bench.get(symbol, [])
                ]
            }
        else:
            return {
                symbol: self.get_status(symbol)
                for symbol in set(list(self.champions.keys()) + list(self.challenger_bench.keys()))
            }

    async def shadow_evaluation_loop(self, arena_manager, stop_event):
        """
        Continuous loop that checks for hot-swap opportunities.
        """
        self.logger.info("[PIT CREW] Shadow evaluation loop started")
        from data_fetcher import get_market_data # Local import for async context

        while not stop_event.is_set():
            try:
                for symbol in self.config.ASSET_UNIVERSE:
                    if not self.challenger_bench.get(symbol):
                        continue

                    self.logger.info(f"[PIT CREW] Running shadow evaluation for {symbol}...")
                    
                    # 1. Fetch recent data for backtesting
                    # Use a longer timeframe for more meaningful shadow trades
                    df = await get_market_data(symbol, '5m', 500, exchange_instance=None)
                    if df is None or df.empty:
                        self.logger.warning(f"Could not fetch data for shadow evaluation of {symbol}")
                        continue

                    # 2. Run shadow backtests for all challengers
                    for challenger in self.challenger_bench.get(symbol, []):
                        await self._run_shadow_backtest(challenger, df)

                    # 3. Check for hot-swap
                    champion_agent = arena_manager.get_agents_for_symbol(symbol)
                    current_champion_id = champion_agent[0].agent_id if champion_agent else None

                    new_champion_id = self.check_for_hotswap(symbol, current_champion_id)

                    if new_champion_id:
                        self.logger.info(f"[PIT CREW] Triggering hot-swap for {symbol}")
                        
                        # Create the new agent
                        from forge.core.agent import V3Agent
                        from forge.crucible.arena_manager import CrucibleAgent
                        new_v3_agent = V3Agent(symbol=symbol, exchange=arena_manager.exchange, app_config=self.config, agent_id=new_champion_id)
                        new_crucible_agent = CrucibleAgent(agent_id=new_champion_id, v3_agent=new_v3_agent)
                        
                        # Perform the swap
                        await arena_manager.replace_agent(symbol, current_champion_id, new_crucible_agent)
                        
                        self.logger.info(f"✅ HOT-SWAP COMPLETE: {new_champion_id} is now the live agent for {symbol}.")

                await asyncio.sleep(300) # Run every 5 minutes

            except asyncio.CancelledError:
                self.logger.info("[PIT CREW] Shadow evaluation loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"[PIT CREW] Error in shadow evaluation loop: {e}", exc_info=True)
                await asyncio.sleep(60)

        self.logger.info("[PIT CREW] Shadow evaluation loop stopped")

    async def _run_shadow_backtest(self, challenger: ChallengerModel, df: pd.DataFrame):
        """Runs a vectorized backtest for a single challenger."""
        from forge.crucible.numba_backtester import NumbaTurboBacktester # VELOCITY UPGRADE
        from forge.data_processing.feature_factory import FeatureFactory

        try:
            self.logger.debug(f"Running shadow backtest for {challenger.model_id}...")
            
            ff = FeatureFactory({challenger.symbol: df.copy()}, logger=self.logger)
            processed_data = ff.create_all_features(app_config=self.config, exchange=None)
            features_df = processed_data.get(challenger.symbol)
            
            if features_df is None or features_df.empty:
                self.logger.warning(f"Feature creation failed for shadow backtest of {challenger.model_id}")
                return

            from forge.armory.model_registry import ModelRegistry
            registry = ModelRegistry(registry_path=self.config.MODEL_DIR)
            model_logic = registry.load_model(challenger.model_id)

            if not model_logic:
                self.logger.error(f"Could not load model logic for challenger {challenger.model_id}")
                return

            # Generate signals from the model logic
            raw_predictions = model_logic.predict(features_df.drop(columns=['open', 'high', 'low', 'close', 'volume'], errors='ignore'))
            signals = np.zeros(len(raw_predictions))
            signals[raw_predictions == 1] = 1 # Entry
            signals[raw_predictions == 2] = 2 # Exit

            initial_capital = self.shadow_state.get(challenger.symbol, {}).get(challenger.model_id, 200.0)
            backtester = NumbaTurboBacktester(initial_capital=initial_capital)
            
            results = backtester.run_backtest(features_df, signals)
            
            metrics = results.get('metrics', {})
            ttt = metrics.get('time_to_target', -1)
            final_capital = results.get('final_capital', initial_capital)
            psr = metrics.get('probabilistic_sharpe_ratio', 0.0)
            drawdown = metrics.get('max_drawdown_pct', 0.0)

            self.record_shadow_performance(challenger.symbol, challenger.model_id, ttt, final_capital, psr, drawdown)

        except Exception as e:
            self.logger.error(f"Error during shadow backtest for {challenger.model_id}: {e}", exc_info=True)

