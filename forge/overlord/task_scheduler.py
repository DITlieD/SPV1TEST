# forge/overlord/task_scheduler.py

# Keep only essential, non-numerical imports at the global level
import os
import asyncio
import traceback
import logging
import joblib
import sys # Added to fix NameError
import dill # Added for potential DNA serialization

# Import utility classes that don't rely heavily on numerical libraries
from forge.utils.pipeline_status import PipelineStatus

# (Remove pandas, numpy, sklearn, LGBMWrapper, FeatureFactory, etc. from global scope)

# Define the main function
def run_single_forge_cycle(raw_data_path: str, asset_symbol: str, reporter: PipelineStatus, app_config, exchange, device: str = "cpu", logger=None, inherited_dna=None, model_instance_id: str = None, shared_state=None):
    """Orchestrates a single, complete, hyper-adaptive model evolution cycle."""
    
    # --- START OF FIX ---
    # Determine which evolution process to run based on context or task type
    # For now, we assume the main bake-off cycle *always* uses the Model Blueprint GA
    EVOLUTION_TYPE = "MODEL_BLUEPRINT_GA" # Explicitly set for clarity
    # --- END OF FIX ---

    try:
        # Delayed Imports within the function
        import pandas as pd
        import numpy as np
        import torch # Added for device check
        import ccxt.pro as ccxt # Added for data fetching

        # Import necessary components
        from data_processing_v2 import get_aligned_mtf_data, run_graph_feature_pipeline # Import data processing
        from forge.data_processing.labeling import get_triple_barrier_labels # FIX: Import corrected function name
        from forge.evolution.feature_synthesizer import evolve_features # FIX: Import the correct function
        from forge.modeling.causal_engine import run_causal_discovery # FIX: Import the correct function
        from forge.modeling.macro_causality import calculate_influence_map # FIX: Import the correct function
        from forge.strategy.environment_classifier import EnvironmentClassifier # Keep for context

        # --- FIX: Import the CORRECT Genetic Algorithm ---
        from forge.blueprint_factory.genetic_algorithm import GeneticAlgorithm
        # --- END FIX ---
        
        # --- FIX: Import the correct fitness function (already used by GA, but good practice) ---
        from forge.crucible.fitness_function import run_fitness_evaluation
        # --- END FIX ---
        
        from forge.evolution.surrogate_manager import SurrogateManager # Keep for SAE
        
        # Ensure imports match file locations
        from validation_gauntlet import run_full_gauntlet # Keep for validation
        
        # Import Model Registry
        from forge.armory.model_registry import ModelRegistry
        
        # Import Models (adjust path if needed)
        from models_v2 import LGBMWrapper, XGBWrapper # Import necessary models
        from forge.models.transformer_wrapper import TransformerWrapper # Import transformer
        # --- End Delayed Imports ---


        logger.info(f"Starting Forge cycle for {asset_symbol} ({model_instance_id})")
        reporter.set_status("Initializing", "Setting up Forge cycle.")
        
        # --- 1. Data Loading and Preparation ---
        reporter.set_status("Data Loading", f"Fetching and aligning MTF data for {asset_symbol}...")
        logger.info("Preparing data...")

        # Initialize exchange within async context manager if needed, or pass one in
        # Reusing the simple init from the workaround for now
        exchange_instance = None
        aligned_data = pd.DataFrame() # Initialize as empty
        loop = None # FIX: Initialize loop to None
        try:
             # This async handling might need refinement depending on the caller context
             exchange_instance = ccxt.bybit({'options': {'defaultType': 'swap'}})
             exchange_instance.set_sandbox_mode(app_config.BYBIT_USE_TESTNET)
             
             # Run the async data fetching function
             # Using asyncio.run() might cause issues if already in an event loop.
             try:
                 loop = asyncio.get_event_loop()
                 if loop.is_running():
                      import nest_asyncio
                      nest_asyncio.apply()
                      logger.warning("Applied nest_asyncio to run async data fetching in task_scheduler.")
                      aligned_data = asyncio.run(get_aligned_mtf_data(asset_symbol, app_config, exchange_instance))
                 else:
                     aligned_data = asyncio.run(get_aligned_mtf_data(asset_symbol, app_config, exchange_instance))
             except RuntimeError as e:
                  logger.error(f"Asyncio runtime error fetching data: {e}. Trying direct run.")
                  aligned_data = asyncio.run(get_aligned_mtf_data(asset_symbol, app_config, exchange_instance))

             if aligned_data.empty:
                  raise ValueError("Failed to get aligned market data.")
             logger.info(f"Aligned data shape: {aligned_data.shape}")

        finally:
             if exchange_instance:
                  try:
                     async def close_exchange_task(): await exchange_instance.close()
                     if loop and loop.is_running(): asyncio.run_coroutine_threadsafe(close_exchange_task(), loop).result(timeout=5)
                     else: asyncio.run(close_exchange_task())
                     logger.info("Exchange closed after data fetching.")
                  except Exception as ex_close: logger.error(f"Error closing exchange: {ex_close}")


        # --- [Optional] Pre-Evolution Steps (Feature Synthesis, CDE) ---
        # These steps modify 'aligned_data' before it's used for training/evolution
        
        # Example: Run Feature Synthesizer (if enabled/needed)
        if app_config.ENABLE_FEATURE_SYNTHESIZER:
             reporter.set_status("Feature Synthesis", "Running Feature Synthesizer...")
             logger.info("[Feature Synthesizer] Starting feature evolution on merged MTFA data...")
             try:
                 # FIX: Call the imported function directly
                 target_series = aligned_data['close'].pct_change().shift(-1).fillna(0)
                 evolved_features_df = evolve_features(
                     df=aligned_data,
                     target=target_series,
                     num_features=5, # Example param
                     generations=30, # Example param
                 )
                 if not evolved_features_df.empty:
                     aligned_data = pd.merge(aligned_data, evolved_features_df, left_index=True, right_index=True, how='left')
                     # Forward fill evolved features to avoid NaNs from merge
                     evolved_cols = evolved_features_df.columns
                     aligned_data[evolved_cols] = aligned_data[evolved_cols].ffill().bfill() # Fill NaNs robustly
                     logger.info(f"[Feature Synthesizer] Successfully injected {len(evolved_cols)} evolved features.")
                 else:
                     logger.warning("[Feature Synthesizer] No evolved features were generated.")
             except Exception as e:
                 logger.error(f"[Feature Synthesizer] Failed: {e}")
                 # Decide whether to continue without evolved features or raise error

        # Example: Run Causal Discovery (if enabled/needed)
        causal_features = None
        if app_config.ENABLE_CAUSAL_DISCOVERY:
             reporter.set_status("Causal Discovery", "Running Causal Discovery Engine...")
             logger.info("[CDE] Running causal discovery...")
             try:
                  # Ensure data has a DatetimeIndex before CDE
                  if not isinstance(aligned_data.index, pd.DatetimeIndex):
                      aligned_data['timestamp'] = pd.to_datetime(aligned_data['timestamp'])
                      aligned_data = aligned_data.set_index('timestamp')

                  # FIX: Call the imported function directly
                  potential_features = [col for col in aligned_data.columns if col not in ['label', 'open', 'high', 'low', 'close', 'volume', 'target_return']]
                  causal_parents = run_causal_discovery(
                      df=aligned_data,
                      var_names=potential_features,
                      target_variable='close'
                  )
                  if causal_parents:
                       causal_features = list(causal_parents)
                       logger.info(f"[CDE] Found {len(causal_features)} causal features: {causal_features}")
                  else:
                       logger.warning("[CDE] Causal discovery returned no features. Proceeding with all features.")
                       causal_features = None # Explicitly set to None
             except Exception as e:
                  logger.error(f"[CDE] Failed: {e}. Proceeding with all features.")
                  causal_features = None # Ensure fallback

        # --- Data Splitting and Final Prep ---
        reporter.set_status("Data Preparation", "Splitting data and creating labels...")
        
        # Labeling (ensure it happens *after* feature synthesis/CDE if they modify data)
        logger.info("Creating labels...")
        atr_col = next((c for c in aligned_data.columns if c.startswith('ATRr')), 'ATRr_14') # Find ATRr column
        if atr_col not in aligned_data.columns:
             raise ValueError(f"Required ATRr column ('{atr_col}' or similar) not found in data.")
             
        labels = get_triple_barrier_labels(
             df=aligned_data,
             pt_sl=app_config.ACN_CONFIG['PT_SL_ATR_MULTIPLE'],
             max_hold_bars=app_config.ACN_CONFIG['LOOKAHEAD_BARS']
        )
        aligned_data['label'] = labels
        # IMPORTANT: Drop rows where labels are NaN (e.g., at the end of the series)
        initial_rows = len(aligned_data)
        aligned_data.dropna(subset=['label'], inplace=True)
        logger.info(f"Labeling complete. Dropped {initial_rows - len(aligned_data)} rows due to NaN labels.")
        
        if aligned_data.empty:
            raise ValueError("Data became empty after adding labels and dropping NaNs.")

        # --- Final Data Sanitization before splitting ---
        aligned_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        aligned_data.dropna(inplace=True)
        if aligned_data.empty:
            raise ValueError("Data became empty after final sanitization.")
        # ---

        # Data Splitting
        train_ratio = 0.7
        val_ratio = 0.2
        n = len(aligned_data)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_data = aligned_data.iloc[:n_train]
        val_data = aligned_data.iloc[n_train : n_train + n_val]
        
        # Define feature columns: Use causal_features if available, else all valid cols
        if causal_features:
            # Ensure causal features actually exist in the dataframe after potential additions/removals
            feature_cols = [f for f in causal_features if f in aligned_data.columns]
            logger.info(f"Using {len(feature_cols)} validated causal features for evolution.")
        else:
            # Fallback: Exclude non-feature columns
            exclude_cols = ['label', 'open', 'close', 'high', 'low', 'volume'] + [c for c in aligned_data.columns if c.startswith('ATRr')]
            feature_cols = [col for col in aligned_data.columns if col not in exclude_cols]
            logger.info(f"Using {len(feature_cols)} non-causal features for evolution.")

        if not feature_cols:
             raise ValueError("No feature columns identified for training/evolution.")

        # Prepare final datasets
        X_train = train_data[feature_cols]
        y_train = train_data['label']
        
        X_val = val_data # Keep non-feature cols needed by fitness_function ('open', 'ATRr_*', 'close' for returns)
        y_val = val_data['label']
        returns_val = val_data['close'].pct_change().fillna(0) # Needed for fitness function

        logger.info(f"Data split complete: X_train {X_train.shape}, X_val {X_val.shape}")
        
        # --- Environment Classification ---
        reporter.set_status("Environment Classification", "Classifying market environment...")
        env_classifier = EnvironmentClassifier()
        # Pass relevant part of the data (e.g., recent validation data)
        current_environment = env_classifier.classify(val_data.tail(100)) # Use recent data
        logger.info(f"[EnvClassifier] Current Environment: {current_environment}")
        reporter.add_metadata("environment", current_environment)


        # --- 3. Evolution / Synthesis ---
        reporter.set_status("Evolution", f"Starting {EVOLUTION_TYPE}...")

        # --- FIX: Instantiate and Run GeneticAlgorithm ---
        if EVOLUTION_TYPE == "MODEL_BLUEPRINT_GA":
            logger.info("[Bake-Off] Training Challenger: GP 2.0 (Model Blueprint GA)...") # Keep log similar?

            # Initialize SurrogateManager
            surrogate_manager = SurrogateManager(
                 X_train=X_train, y_train=y_train,
                 X_val=X_val, y_val=y_val, returns_val=returns_val,
                 num_processes=max(1, app_config.NUM_PARALLEL_WORKERS // 2), # Use fewer for surrogate?
                 logger=logger
            )

            # Deserialize inherited DNA if it exists (should be a blueprint dict now)
            inherited_blueprint_dict = None
            if inherited_dna:
                try:
                    inherited_blueprint_dict = dill.loads(inherited_dna)
                    # Basic validation: check if it looks like a blueprint dict
                    if not isinstance(inherited_blueprint_dict, dict) or 'architecture' not in inherited_blueprint_dict:
                         logger.warning("Inherited DNA did not seem to be a valid ModelBlueprint dictionary. Starting fresh.")
                         inherited_blueprint_dict = None
                    else:
                         logger.info("Successfully deserialized inherited ModelBlueprint DNA.")
                except Exception as e:
                    logger.error(f"Failed to deserialize inherited DNA as ModelBlueprint: {e}. Starting fresh.")
                    inherited_blueprint_dict = None


            ga = GeneticAlgorithm(
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val, returns_val=returns_val,
                min_horizon=app_config.GA_MIN_HORIZON,
                max_horizon=app_config.GA_MAX_HORIZON,
                population_size=app_config.GA_POPULATION_SIZE,
                generations=app_config.GA_GENERATIONS,
                surrogate_manager=surrogate_manager, # Pass the surrogate manager
                use_surrogate=app_config.USE_SURROGATE_ASSISTANCE, # Control via config
                num_processes=app_config.NUM_PARALLEL_WORKERS,
                logger=logger,
                champion_blueprint=inherited_blueprint_dict, # Pass deserialized dict
                device=device
                # Add other GA params from config if needed
            )

            hall_of_fame = ga.run() # Returns list of best ModelBlueprint objects

            if not hall_of_fame:
                 logger.warning("[Bake-Off] Model Blueprint GA failed to find any valid strategies.")
                 challengers = [] # No challengers if GA failed
            else:
                 # Prepare challengers list for the gauntlet
                 # The gauntlet needs a model object, not just the blueprint
                 challengers = []
                 for bp in hall_of_fame[:app_config.GA_HALL_OF_FAME_SIZE]: # Take top N from HoF
                     try:
                          ModelClass = ga.model_map.get(bp.architecture)
                          if ModelClass:
                               # Simple wrapper for gauntlet compatibility
                               class BlueprintGauntletWrapper:
                                   def __init__(self, model_cls, blueprint):
                                       self._model = model_cls(**blueprint.hyperparameters)
                                       self.blueprint = blueprint
                                       self.features = list(blueprint.features) # Gauntlet needs features
                                   def fit(self, X, y): self._model.fit(X[self.features], y)
                                   def predict(self, X): return self._model.predict(X[self.features])
                                   # Add get_params if gauntlet needs it
                                   def get_params(self, deep=True):
                                        return {"architecture": self.blueprint.architecture,
                                                "features": self.features,
                                                **self.blueprint.hyperparameters,
                                                **self.blueprint.risk_parameters}

                               challengers.append({
                                   "name": f"GP2_{bp.architecture}",
                                   "model": BlueprintGauntletWrapper(ModelClass, bp),
                                   "dna": dill.dumps(vars(bp)), # Serialize blueprint dict as DNA
                                   "explanation": f"Evolved {bp.architecture} model via GA.",
                                   "type": "evolved_blueprint" # Add type identifier
                               })
                          else:
                               logger.warning(f"Could not find model class for HoF blueprint: {bp.architecture}")
                     except Exception as e:
                          logger.error(f"Error preparing challenger from blueprint {bp}: {e}")
                 logger.info(f"Prepared {len(challengers)} challengers from GA Hall of Fame.")

        # --- FIX: Remove the old StrategySynthesizer call ---
        # else: # If EVOLUTION_TYPE was something else (e.g., "STRATEGY_SYNTHESIZER")
        #     logger.info("[Bake-Off] Training Challenger: GP (Strategy Synthesizer)...")
        #     synthesizer = StrategySynthesizer(...) # Original code would go here
        #     best_individual = synthesizer.run()
        #     # Need to adapt the output (best_individual) for the gauntlet
        #     # This part needs careful implementation if StrategySynthesizer is still needed
        #     challengers = [...] # Prepare challengers list based on synthesizer output
        # --- END FIX ---


        # --- 4 & 5. Bake-Off & Gauntlet --- (Assuming these run on the 'challengers' list)
        reporter.set_status("Validation", "Running Bake-Off and Validation Gauntlet...")
        
        # --- FIX: Adapt Gauntlet call ---
        # The existing code expects 'challengers' to be a list of dicts with 'name', 'model', 'dna', etc.
        # We prepared this list above if the GA was successful.

        if not challengers:
             logger.warning("No challengers generated from evolution. Skipping Gauntlet.")
             gauntlet_results = []
        else:
             logger.info(f"Starting Gauntlet for {len(challengers)} challengers...")
             # Pass the full X_val which includes 'open', 'close', 'ATRr_*' etc.
             gauntlet_results = []
             for challenger in challengers:
                 try:
                      logger.info(f"--- Validating Challenger: {challenger['name']} ---")
                      results = run_full_gauntlet(
                           model=challenger["model"], # The wrapped model instance
                           X_train=X_train, y_train=y_train, # Gauntlet might retrain on full train+val? Adjust if needed.
                           X_val=X_val, y_val=y_val, # Pass the validation data
                           symbol=asset_symbol,
                           logger=logger
                           # Add other gauntlet params as needed
                      )
                      gauntlet_results.append({
                           **challenger, # Keep original info like name, dna
                           "results": results # Add gauntlet output
                      })
                 except Exception as e:
                      logger.error(f"Error running Gauntlet for {challenger['name']}: {e}")
                      # Optionally add failed result entry
                      gauntlet_results.append({**challenger, "results": {"error": str(e), "all_passed": False}})

        # --- 6. Champion Selection ---
        reporter.set_status("Selection", "Selecting champion from Gauntlet results...")
        logger.info(f"\n{'='*60}\\n| FORGE: Selecting Champion from Gauntlet Results\\n{'='*60}")
        
        # Filter for challengers that passed all gauntlet steps
        passed_challengers = [c for c in gauntlet_results if c.get("results", {}).get("passed_all_steps", False)] # Updated key access

        if not passed_challengers:
            logger.info("\n[FORGE] ❌ No models survived the gauntlet.")
            reporter.set_status("Finished", "No models passed the validation gauntlet.")
            return None, None # Return None, None to indicate no champion

        # Select the best based on a primary metric (e.g., Sharpe ratio from WFA)
        # Adjust metric key based on actual gauntlet output structure
        try:
             # Example: Prioritize Sharpe from walk-forward analysis if available
             champion_info = max(
                 passed_challengers,
                 key=lambda c: c.get("results", {}).get("metrics", {}).get("wfa_metrics", {}).get("sharpe_ratio", -np.inf)
             )
             logger.info(f"[FORGE] ✅ Champion Selected: {champion_info['name']}")
        except ValueError: # Handle cases where metrics might be missing
             logger.warning("Could not determine champion based on WFA Sharpe. Selecting first passed challenger.")
             champion_info = passed_challengers[0] # Fallback


        # --- 7. Model Registration ---
        reporter.set_status("Registration", f"Registering champion model: {champion_info['name']}")
        registry = ModelRegistry()
        
        # Prepare blueprint/params for registration
        # If champion came from GA, its 'model' is BlueprintGauntletWrapper
        registered_params = {}
        if isinstance(champion_info.get("model"), BlueprintGauntletWrapper):
            # Extract blueprint info
            bp = champion_info["model"].blueprint
            registered_params = {
                "architecture": bp.architecture,
                "features": list(bp.features), # Convert frozenset to list for JSON
                "hyperparameters": bp.hyperparameters,
                "training_horizon": bp.training_horizon,
                "risk_parameters": bp.risk_parameters,
                "ga_fitness": bp.fitness # Include fitness it achieved in GA
            }
        else:
             # Handle champions from other sources (MetaLearner, etc.) if they exist
             # Need to get their relevant parameters
             registered_params = {"info": "Parameters not extracted for non-GA champion"}


        model_id = registry.register_model(
             symbol=asset_symbol,
             model_instance_id=model_instance_id,
             model_type=champion_info['name'], # Use challenger name as type?
             model_params=registered_params, # Store extracted blueprint/params
             performance_metrics=champion_info.get("results", {}).get("metrics", {}), # Store gauntlet metrics
             # model_object=None # Don't store the trained model object itself in registry
             # xai_report_path=None # Add path if XAI is generated
        )
        
        # Get the DNA (serialized blueprint dict) associated with the champion
        winning_dna = champion_info.get("dna") # This should be the serialized dict from GA challenger prep

        logger.info(f"Champion registered with ID: {model_id}")
        return model_id, winning_dna # Return ID and DNA

    except ImportError as e:
        error_msg = f"Failed to import necessary modules inside Forge cycle: {e}\n{traceback.format_exc()}" # Add traceback
        reporter.set_status("Error", error_msg)
        if logger: logger.error(error_msg)
        print(error_msg, file=sys.stderr) # Print to stderr
        return None, None
    except Exception as e:
        error_msg = f"Forge cycle failed: {e}\n{traceback.format_exc()}" # Add traceback
        reporter.set_status("Error", error_msg)
        if logger: logger.error(error_msg)
        print(error_msg, file=sys.stderr) # Print to stderr
        return None, None