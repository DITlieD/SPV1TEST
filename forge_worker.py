# forge_worker.py

import os
import logging
import sys
import time

# Delay complex imports until inside the function.

def run_forge_process(symbol, model_instance_id, generation, serialized_dna=None, shared_state=None):
    try:
        # --- CRITICAL DEADLOCK FIX: Enforce Single Threading ---
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'

        worker_pid = os.getpid()
        
        # --- CRITICAL LOGGING ISOLATION FIX ---
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        if root_logger.hasHandlers():
            root_logger.handlers.clear()

        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(f'%(asctime)s - [Wkr {worker_pid}] - %(name)s - %(message)s')
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

        logger = logging.getLogger("ForgeWorkerMain")

        # Delayed Imports
        import dill
        import torch
        import config as app_config
        from forge.overlord.task_scheduler import run_single_forge_cycle
        from forge.utils.pipeline_status import PipelineStatus

        # Deserialize DNA
        inherited_dna = None
        if serialized_dna:
            inherited_dna = dill.loads(serialized_dna)
            logger.info("Successfully deserialized inherited DNA.")

        reporter = PipelineStatus()
        
        timeframe = app_config.TIMEFRAMES['tactical']
        base_symbol = symbol.split(':')[0].replace('/', '')
        raw_data_path = os.path.join("data", f"{base_symbol}_{timeframe}_raw.csv")

        enforced_device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Forge worker for {symbol} will use device: {enforced_device}")
        
        model_id, winning_dna = run_single_forge_cycle(
            raw_data_path=raw_data_path, asset_symbol=symbol, reporter=reporter,
            app_config=app_config, exchange=None, device=enforced_device,
            logger=logger, inherited_dna=inherited_dna, model_instance_id=model_instance_id,
            shared_state=shared_state
        )
        
        logger.info(f"✅ Forge cycle for {model_instance_id} completed.")
        
        serialized_winning_dna = None
        if winning_dna:
            serialized_winning_dna = dill.dumps(winning_dna)

        return symbol, model_id, serialized_winning_dna
        
    except Exception as e:
        # Broad exception handler to catch anything that might kill the worker
        # Use a temporary logger in case the main one failed
        temp_logger = logging.getLogger("ForgeWorkerRecovery")
        temp_logger.error(f"FATAL ERROR in subprocess forge for {model_instance_id}: {e}", exc_info=True)
        return symbol, None, None