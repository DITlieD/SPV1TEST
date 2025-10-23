# singularity_engine.py
import os
import pandas as pd
from datetime import datetime, timedelta
import logging
import asyncio
import json
import numpy as np
import time
import functools

from forge.utils.pipeline_status import PipelineStatus
from forge.monitoring.drift_detector import DriftDetector
from forge.models.ensemble_manager import HedgeEnsembleManager
from forge.acp.acp_engines import ChimeraEngine
from forge.evolution.opponent_templates import simple_rsi_opponent, opponent_param_bounds

class SingularityEngine:
    def __init__(self, trigger_trader_restart_func, trigger_risk_reduction_func, trigger_weight_update_func, status_reporter: PipelineStatus, stop_event: asyncio.Event, app_config, hedge_manager,
                 logger=None, submit_forge_task_func=None, shared_state=None):
        
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger("SingularityFallback")
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False
        
        self.submit_forge_task = submit_forge_task_func
        self.trigger_trader_rescan = trigger_trader_restart_func
        self.trigger_risk_reduction = trigger_risk_reduction_func
        self.trigger_weight_update_func = trigger_weight_update_func
        self.reporter = status_reporter
        self.stop_event = stop_event
        self.config = app_config
        self.device = self.config.DEVICE
        self.shared_state = shared_state
        # VELOCITY UPGRADE: Aggressive CWO - Run reactive loop every 60s instead of 600s
        self.drift_check_interval = 60

        self.drift_detectors = {symbol: DriftDetector() for symbol in self.config.ASSET_UNIVERSE}
        self.last_log_timestamp = None
        
        self.asset_queue = asyncio.Queue()
        for asset in self.config.ASSET_UNIVERSE:
            self.asset_queue.put_nowait(asset)

        self.hedge_manager = hedge_manager
        self.active_forges = set()
        self.cycle_completion_event = asyncio.Event()

        # --- ACP Integration: Initialize Chimera Engine ---
        self.opponent_param_bounds = opponent_param_bounds
        self.chimera_engine = ChimeraEngine(
            strategy_template=simple_rsi_opponent,
            max_iterations=50 # Configurable
        )
        # --- End ACP Integration ---

        self.logger.info("Singularity Engine initialized.")
        self.logger.info(f"Asset universe order: {self.config.ASSET_UNIVERSE}")

    async def _proactive_forge_loop(self):
        """
        UPDATE V5: Schedules evolution for each specific model instance defined in DEPLOYMENT_STRATEGY.
        Creates unique model instances (4 BTC, 2 ETH, 2 SOL) that evolve independently.
        """
        self.logger.info("Starting continuous, multi-model Forge loop (Update V5)...")
        PRIORITY_LOW = 10

        # Get deployment strategy from config
        deployment_strategy = getattr(self.config, 'DEPLOYMENT_STRATEGY', {})

        if not deployment_strategy:
            self.logger.error("DEPLOYMENT_STRATEGY is empty or missing in config. Halting Forge loop.")
            return

        total_models = sum(deployment_strategy.values())
        self.logger.info(f"Deployment Strategy: {total_models} models across {len(deployment_strategy)} assets")

        while not self.stop_event.is_set():
            # Wait for all tasks from the previous cycle to complete before starting a new one.
            if self.active_forges:
                self.logger.info(f"Waiting for {len(self.active_forges)} active forge task(s) to complete from previous cycle...")
                await self.cycle_completion_event.wait()

            self.logger.info(f"Starting a new forging cycle for {total_models} model instances...")
            self.cycle_completion_event.clear() # Reset the event for the new cycle.

            tasks_submitted_this_cycle = 0

            # UPDATE V5: Iterate through deployment strategy
            for symbol, count in deployment_strategy.items():
                if self.stop_event.is_set():
                    break

                # Create multiple model instances per asset
                for i in range(count):
                    if self.stop_event.is_set():
                        break

                    # Create unique identifier for this specific model instance
                    # Example: BTCUSDT_Model_1, BTCUSDT_Model_2, etc.
                    sanitized_symbol = symbol.replace('/', '').replace(':', '')
                    model_instance_id = f"{sanitized_symbol}_Model_{i+1}"

                    # Check if this instance is already being trained
                    if model_instance_id in self.active_forges:
                        self.logger.debug(f"Model instance {model_instance_id} is already active. Skipping.")
                        continue

                    try:
                        self.logger.info(f"[Proactive Forge] Submitting task for {symbol} (Instance {i+1}/{count}). ID: {model_instance_id}")
                        self.active_forges.add(model_instance_id)
                        tasks_submitted_this_cycle += 1

                        done_callback = functools.partial(self._on_forge_task_done, model_instance_id)

                        # UPDATE V5: Pass model_instance_id and enable DNA inheritance
                        # For first-time models, no elite DNA will exist (starts from scratch)
                        # For re-forging, elite DNA from previous successful models will be inherited
                        self.logger.info("Calling self.submit_forge_task")
                        await self.submit_forge_task(
                            symbol=symbol,
                            priority=PRIORITY_LOW,
                            source="Proactive",
                            inherit_dna=True,  # Enable elite DNA inheritance
                            model_instance_id=model_instance_id,
                            done_callback=done_callback
                        )

                    except Exception as e:
                        self.logger.error(f"Error submitting Forge task for {model_instance_id}: {e}", exc_info=True)
                        self.active_forges.discard(model_instance_id)

            if tasks_submitted_this_cycle > 0:
                self.logger.info(f"Submitted {tasks_submitted_this_cycle} forge tasks for this cycle. Now waiting for completion...")
                await self.cycle_completion_event.wait()

            self.logger.info("Completed a full forging cycle. Waiting before next cycle...")
            await asyncio.sleep(10)

    def _on_forge_task_done(self, model_instance_id):
        """
        UPDATE V5: Callback to remove a model instance from the active forges set.
        Called when a specific model instance completes training.
        """
        self.logger.info(f"Forge task for {model_instance_id} completed. Removing from active set.")
        self.active_forges.discard(model_instance_id)
        
        if not self.active_forges:
            self.logger.info("All active forge tasks for the cycle are complete.")
            self.cycle_completion_event.set()

    async def _run_chimera_inference(self):
        """Periodically runs Chimera to infer opponent parameters."""
        from forge.acp.acp_engines import extract_observed_market_signature
        from data_fetcher import get_market_data
        
        if self.chimera_engine:
            try:
                # Use BTC/USDT as the representative asset
                symbol = 'BTC/USDT'
                df = await get_market_data(symbol, '1h', 500)
                if df is None or df.empty:
                    self.logger.warning(f"Could not fetch data for Chimera inference for {symbol}")
                    return

                observed_market_signature = extract_observed_market_signature(df)

                inference_result = await asyncio.get_running_loop().run_in_executor(
                    None,
                    self.chimera_engine.infer_strategy_parameters,
                    observed_market_signature,
                    df,
                    self.opponent_param_bounds
                )
                if inference_result and inference_result.get('fit_quality', 0) > 0.5:
                    if self.shared_state is not None:
                        self.shared_state['inferred_opponent_params'] = inference_result['parameters']
                    self.logger.info(f"Chimera inferred opponent params: {inference_result['parameters']} (Fit: {inference_result['fit_quality']:.2f})")
                else:
                    self.logger.warning("Chimera inference failed or fit quality too low.")
            except Exception as chimera_e:
                self.logger.error(f"Error during Chimera inference: {chimera_e}")

    async def _chimera_loop(self):
        """Periodically runs Chimera inference.""" 
        while not self.stop_event.is_set():
            await self._run_chimera_inference()
            await asyncio.sleep(3600) # Run every hour
    async def run(self):
        self.logger.info("Singularity Engine starting its operational loops.")
        tasks = [
            self._proactive_forge_loop(),
            self._chimera_loop(),
        ]
        await asyncio.gather(*tasks)