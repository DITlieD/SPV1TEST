# forge/overlord/task_scheduler.py

# Keep only essential, non-numerical imports at the global level
import os
import asyncio
import traceback
import logging

# Import utility classes that don't rely heavily on numerical libraries
from forge.utils.pipeline_status import PipelineStatus

# (Remove pandas, numpy, sklearn, LGBMWrapper, FeatureFactory, etc. from global scope)

# Define the main function
def run_single_forge_cycle(raw_data_path: str, asset_symbol: str, reporter: PipelineStatus, app_config, exchange, device: str = "cpu", logger=None, inherited_dna=None, model_instance_id: str = None, shared_state=None):
    """Orchestrates a single, complete, hyper-adaptive model evolution cycle."""
    
    try:
        import pandas as pd
        import numpy as np
        # Import all necessary components that were previously global
        from forge.data_processing.feature_factory import FeatureFactory
        from forge.data_processing.labeling import create_categorical_labels
        from forge.evolution.strategy_synthesizer import StrategySynthesizer
        from deap import gp 
        # Ensure imports match file locations (assuming validation_gauntlet is in the main folder)
        from validation_gauntlet import run_full_gauntlet, EvolvedStrategyWrapper
        from forge.armory.model_registry import ModelRegistry
        # Assuming models_v2 is in the main folder based on context
        from models_v2 import LGBMWrapper 
        # TransformerWrapper might also need delayed import if it imports torch/numpy globally
        from forge.models.transformer_wrapper import TransformerWrapper
        from forge.crucible.backtester import VectorizedBacktester
        from forge.crucible.numba_backtester import NumbaTurboBacktester
        from forge.monitoring.ai_analyst import get_strategy_explanation
        from forge.strategy.environment_classifier import EnvironmentClassifier
        # VELOCITY UPGRADE: Meta-Learning Imports
        from forge.models.meta_learner import MetaLearner, ReptileTrainer
        from forge.modeling.causal_engine import run_causal_discovery
        from forge.evolution.feature_synthesizer import evolve_features
        # NOTE: AdversarialForge removed - using StrategySynthesizer.run() directly for performance
        import torch
        import joblib

        if logger is None:
            logger = logging.getLogger(__name__)

        # --- MTFA DATA LOADING: Load multiple timeframes ---
        reporter.set_status("Data Loading (MTFA)", f"Loading multi-timeframe data for {asset_symbol}...")

        mtfa_data = {}
        base_symbol = asset_symbol.split(':')[0].replace('/', '')

        # Define data limits for each timeframe (optimized for ULTRA MFT)
        def get_mtfa_limit(tf_name):
            try:
                seconds = pd.to_timedelta(tf_name).total_seconds()
            except ValueError:
                return 1000  # Default fallback

            if seconds <= 60: return 2500    # 1m: ~41 hours of recent data
            if seconds <= 900: return 500    # 15m: ~5 days of data
            return 500                       # 1h: ~20 days of data

        # Load all required timeframes defined in config
        logger.info(f"[MTFA] Loading timeframes: {list(app_config.TIMEFRAMES.values())}")

        for tf_role, tf_name in app_config.TIMEFRAMES.items():
            # Build path to pre-fetched CSV data (in Forge worker context)
            tf_path = f"data/{base_symbol}_{tf_name}_raw.csv"

            if os.path.exists(tf_path):
                limit = get_mtfa_limit(tf_name)
                df = pd.read_csv(tf_path, index_col='timestamp', parse_dates=True).tail(limit)
                mtfa_data[tf_name] = df
                logger.info(f"[MTFA] Loaded {tf_name}: {len(df)} bars from {tf_path}")
            else:
                logger.warning(f"[MTFA] Missing data for {tf_name} at {tf_path}")
                # Try using the main raw_data_path if it's 1m data
                if tf_name == '1m' and os.path.exists(raw_data_path):
                    df = pd.read_csv(raw_data_path, index_col='timestamp', parse_dates=True).tail(2500)
                    mtfa_data[tf_name] = df
                    logger.info(f"[MTFA] Loaded {tf_name} from raw_data_path: {len(df)} bars")

        # Verify we have at least the microstructure (1m) data
        ltf_name = app_config.TIMEFRAMES['microstructure']
        if ltf_name not in mtfa_data or mtfa_data[ltf_name].empty:
            raise FileNotFoundError(f"Primary LTF data ({ltf_name}) missing for {asset_symbol}")

        logger.info("[MTFA] Starting multi-timeframe processing and alignment...")

        # Use the centralized MTFA utility (exchange is None in Forge worker context)
        from data_processing_v2 import process_and_align_mtfa
        df_full_features = process_and_align_mtfa(mtfa_data, app_config, exchange=None)

        logger.info(f"[MTFA] Processing complete: {len(df_full_features)} bars with {len(df_full_features.columns)} features")

        if df_full_features.empty:
            raise ValueError("MTFA processing resulted in an empty DataFrame.")

        # --- 2.5. Feature Synthesizer (God Protocol - Pillar II) ---
        # Run ONCE on the final merged MTFA data (not on intermediate timeframes)
        reporter.set_status("Feature Synthesis", "Evolving proprietary features...")
        logger.info("[Feature Synthesizer] Starting feature evolution on merged MTFA data...")

        target = df_full_features['close'].pct_change().shift(-1).fillna(0)
        evolved_features_df = evolve_features(df_full_features, target, num_features=5)

        if not evolved_features_df.empty:
            logger.info(f"[Feature Synthesizer] Injecting {len(evolved_features_df.columns)} evolved features...")
            # Merge evolved features into main dataframe
            for col in evolved_features_df.columns:
                df_full_features[col] = evolved_features_df[col]
            logger.info(f"[Feature Synthesizer] Successfully injected evolved features: {list(evolved_features_df.columns)}")
        else:
            logger.warning("[Feature Synthesizer] No evolved features generated")

        # --- 3. Causal Discovery Engine (Pillar I) ---
        reporter.set_status("Causal Discovery", "Identifying causal features...")

        # Define the target variable for causal discovery (e.g., 'close' as a proxy for returns)
        causal_target = 'close'

        # Get the list of all potential features to analyze
        # CRITICAL: Exclude target_return and label to prevent data leakage
        excluded_cols = ['open', 'high', 'low', 'volume', 'target_return', 'label']
        potential_features = [col for col in df_full_features.columns if col not in excluded_cols]
        
        # Run causal discovery
        causal_features = run_causal_discovery(
            df=df_full_features,
            var_names=potential_features,
            target_variable=causal_target
        )

        if not causal_features:
            logger.warning("[CDE] Causal discovery returned no features. Proceeding with all features.")
            # If no causal features are found, use all features except the core ones and leakage columns
            causal_features = set([col for col in potential_features if col not in ['open', 'high', 'low', 'close', 'volume', 'target_return', 'label']])

        logger.info(f"[CDE] Using {len(causal_features)} causally-informed features for evolution.")
        reporter.log(f"Causal features identified: {len(causal_features)}")

        # --- 4. Labeling and Data Splitting ---
        reporter.set_status("Labeling", "Creating triple-barrier labels...")
        df_full_features['label'] = create_categorical_labels(df_full_features)
        
        df_full_features.dropna(inplace=True)

        # --- DATA SANITIZATION CHECKPOINT ---
        numeric_features = df_full_features.select_dtypes(include=np.number)
        if not np.isfinite(numeric_features.values).all():
             raise ValueError("Persistent data corruption (NaN/Inf) detected. Halting Forge cycle.")
        
        train_size = int(len(df_full_features) * 0.8)
        
        df_train_full = df_full_features.iloc[:train_size].copy()
        df_gauntlet_full = df_full_features.iloc[train_size:].copy()
        df_train_numeric = df_train_full.select_dtypes(include=np.number).copy()
        df_gauntlet_numeric = df_gauntlet_full.select_dtypes(include=np.number)

        # --- 3. Environment Classification ---
        reporter.set_status("Environment Analysis", "Classifying market environment...")
        env_classifier = EnvironmentClassifier()
        market_environment = env_classifier.classify(df_train_numeric)
        reporter.log(f"Market environment classified as: {market_environment}")

        # --- DEFINITIVE FIX: Define feature_names before the Bake-Off ---
        feature_names = [col for col in df_train_numeric.columns if col in causal_features]

        # --- 4. The Bake-Off: Train All Challengers ---
        reporter.set_status("Bake-Off", "Training all challenger models...")
        print(f"\n{'='*60}\n| FORGE: Starting Bake-Off for {asset_symbol} in '{market_environment}'\n{'='*60}")
        challengers = []

        class MetaLearnerWrapper:
            """A wrapper to make the PyTorch MetaLearner compatible with the backtester."""
            def __init__(self, model, features, device='cpu', model_params=None):
                self.model = model.to(device)
                self.features = features
                self.device = device
                self.model.eval()
                self.is_meta_learner = True
                self.model_params = model_params or {}

            def get_dna(self):
                return {
                    "architecture": "MetaLearner",
                    "feature_names": self.features,
                    "model_params": self.model_params
                }

            def predict(self, df: pd.DataFrame) -> np.ndarray:
                if not all(f in df.columns for f in self.features):
                    missing = set(self.features) - set(df.columns)
                    raise ValueError(f"Missing features for MetaLearner prediction: {missing}")
                
                X = torch.tensor(df[self.features].values, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    probs = self.model(X)
                    predictions = torch.argmax(probs, dim=1).cpu().numpy()
                return predictions

            def fast_update(self, df_recent: pd.DataFrame, lr: float = 0.01, steps: int = 3):
                """Fine-tunes the model on recent data for rapid adaptation."""
                if 'label' not in df_recent.columns:
                    return
                
                self.model.train()
                optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
                loss_fn = torch.nn.CrossEntropyLoss()

                X = torch.tensor(df_recent[self.features].values, dtype=torch.float32).to(self.device)
                y = torch.tensor(df_recent['label'].values, dtype=torch.long).to(self.device)

                for _ in range(steps):
                    optimizer.zero_grad()
                    predictions = self.model(X)
                    loss = loss_fn(predictions, y)
                    loss.backward()
                    optimizer.step()
                
                self.model.eval()

        def train_gp_islands(strategy_type='reversion'):
            """Train GP 2.0 challenger"""
            try:
                POP_SIZE = 2001
                GENERATIONS = 101

                reporter.set_status("GP 2.0", f"Evolving (Pop: {POP_SIZE}, Gen: {GENERATIONS}, Type: {strategy_type})...")
                print(f"\n[Bake-Off] Training Challenger: GP 2.0 (Pop: {POP_SIZE}, Gen: {GENERATIONS}, Type: {strategy_type})...")

                # Define inputs for GP (numeric features, excluding OHLCV/Label/Target)
                # CRITICAL: Exclude target_return to prevent data leakage!
                X_gp_input = df_train_numeric.drop(columns=['label', 'open', 'high', 'low', 'close', 'volume', 'target_return'], errors='ignore')

                # Full dataframe for backtesting context (needs OHLCV)
                X_fitness_context = df_train_full
                y_fitness = df_train_full['label']

                feature_names_gp = X_gp_input.columns.tolist()

                # Counter for logging first evaluation and errors
                first_eval_logged = [False]
                first_error_logged = [False]

                def gp_fitness_evaluator(strategy_logic_func, adversarial_scenario=None):
                    try:
                        # Apply adversarial scenario if provided
                        fitness_context = X_fitness_context.copy()
                        if adversarial_scenario:
                            fitness_context = adversarial_scenario(fitness_context)

                        # Apply evolved logic (pandas apply is actually optimized)
                        raw_predictions = X_gp_input.apply(lambda row: strategy_logic_func(*row), axis=1)

                        # Convert to entry/exit signals with state tracking
                        signals = np.zeros(len(raw_predictions), dtype=int)
                        in_position = False
                        for i in range(len(raw_predictions)):
                            if raw_predictions.iloc[i]: # Strategy says "be in position"
                                if not in_position:
                                    signals[i] = 1  # Entry
                                    in_position = True
                            else: # Strategy says "be out of position"
                                if in_position:
                                    signals[i] = 2  # Exit
                                    in_position = False
                        
                        # CORE FIX: Use the high-fidelity backtester for fitness evaluation
                        backtester = NumbaTurboBacktester()
                        
                        results = backtester.run_backtest(fitness_context, signals)
                        metrics = results['metrics']

                        # --- V9.4 FIX: Define variables before logging ---
                        sharpe = metrics.get('sharpe_ratio', -10.0)
                        profit_factor = metrics.get('profit_factor', 0.0)
                        # ---

                        # --- V9.3 DEFINITIVE FITNESS ALIGNMENT ---
                        # The WFA uses 'ttt_fitness' and is successful. The main evolution MUST use the same metric.
                        # This aligns the entire system to one definition of "good".
                        metrics['adaptive_fitness'] = metrics.get('ttt_fitness', -1e9)

                        # Also, enforce the minimum trades rule consistently.
                        if metrics.get('total_trades', 0) < 10:
                            metrics['adaptive_fitness'] = -100.0
                        # ---

                        # DIAGNOSTIC: Log first evaluation
                        if not first_eval_logged[0]:
                            logger.info(f"[GP Eval FIRST] Adaptive Fitness={metrics.get('adaptive_fitness', -1e9):.4f}, "
                                      f"Sharpe={sharpe:.2f}, Profit Factor={profit_factor:.2f}, "
                                      f"Trades={metrics.get('total_trades', 0)}")
                            first_eval_logged[0] = True

                        # CRITICAL DIAGNOSTIC: Log suspicious high-fitness individuals
                        if metrics.get('adaptive_fitness', -1e9) > 10: # A high but more realistic threshold
                            logger.warning(f"[SUSPICIOUS FITNESS] Fitness={metrics.get('adaptive_fitness', 0):.2f}, "
                                         f"PnL={metrics.get('total_pnl_pct', 0):.2f}%, "
                                         f"Trades={metrics.get('total_trades', 0)}, "
                                         f"Sharpe={metrics.get('sharpe_ratio', 0):.2f}, "
                                         f"WinRate={metrics.get('win_rate', 0):.2f}")
                            # Log first 20 signals to see the pattern
                            signal_pattern = [int(s) for s in signals[:20]]
                            logger.warning(f"[SUSPICIOUS] Signal pattern (first 20): {signal_pattern}")

                        # Return full metrics dict for strategy_synthesizer to extract fitness
                        return metrics, signals, fitness_context

                    except Exception as e:
                        if not first_error_logged[0]:
                            logger.error(f"[GP Eval FIRST ERROR] {type(e).__name__}: {e}", exc_info=True)
                            first_error_logged[0] = True
                        # Return a metrics dict with bad fitness on error
                        return {'adaptive_fitness': -10.0, 'total_trades': 0, 'total_pnl_pct': -100.0}, np.array([]), pd.DataFrame()

                # ELITE DNA INHERITANCE: Use DNA passed from crucible_engine (already retrieved from ElitePreservationSystem)
                # UPDATE V5: inherited_dna is already deserialized by forge_worker
                if inherited_dna:
                    logger.info(f"[GP 2.0] 🧬 Elite DNA received for {model_instance_id}! Seeding evolution with proven winner.")
                else:
                    logger.info(f"[GP 2.0] No elite DNA provided for {model_instance_id}. Starting from scratch.")

                # Save fitness context for worker processes
                context_path = f"temp_fitness_context_{model_instance_id}.joblib"
                joblib.dump(X_fitness_context, context_path)

                # Initialize synthesizer with elite DNA
                synthesizer = StrategySynthesizer(
                    feature_names_gp,
                    gp_fitness_evaluator,
                    population_size=POP_SIZE,
                    generations=GENERATIONS,
                    logger=logger,
                    seed_dna=inherited_dna,  # ELITE INHERITANCE!
                    reporter=reporter,
                    asset_symbol=asset_symbol,
                    shared_state=shared_state,
                    fitness_context_path=context_path
                )

                # FIX: Use StrategySynthesizer.run() instead of AdversarialForge
                # (Bypasses sequential bottleneck, uses Loky optimized SAE)
                logger.info("[GP 2.0] Running StrategySynthesizer (SAE/Loky Optimized)...")
                try:
                    best_strategy_tree = synthesizer.run()
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    logger.error(f"[GP 2.0] StrategySynthesizer crashed: {e}", exc_info=True)
                    raise e

                if best_strategy_tree and best_strategy_tree.fitness.valid:
                    evolved_model = EvolvedStrategyWrapper(best_strategy_tree, synthesizer.pset, synthesizer.feature_names)
                    explanation = get_strategy_explanation(best_strategy_tree, synthesizer.pset)
                    dna = evolved_model.get_dna() if hasattr(evolved_model, 'get_dna') else None

                    print(f"[Bake-Off] GP 2.0 training complete. Best Fitness: {best_strategy_tree.fitness.values[0]:.4f}")
                    reporter.log(f"[Bake-Off] GP 2.0 Evolved. Fitness: {best_strategy_tree.fitness.values[0]:.4f}")

                    return {"name": "GP 2.0", "model": evolved_model, "explanation": explanation, "dna": dna}
                else:
                    print("[Bake-Off] GP 2.0 failed to find a valid strategy.")
                    return None

            except Exception as e:
                print(f"[Bake-Off] ERROR: GP 2.0 training failed: {e}")
                logger.error(f"[Bake-Off] ERROR: GP 2.0 training failed: {e}", exc_info=True)
                return None

        def train_meta_learner():
            """Train Meta-Learner challenger"""
            try:
                print(f"\n[Bake-Off] Training Challenger: Meta-Learner...")
                reporter.set_status("Bake-Off: Meta-Learner", "Training adaptive neural model...")

                # Prepare data for Meta-Learner
                X_meta = df_train_numeric[feature_names].values
                y_meta = df_train_numeric['label'].values

                # Initialize Meta-Learner
                input_dim = len(feature_names)
                meta_model = MetaLearner(input_dim=input_dim, hidden_dim=64, output_dim=3)
                trainer = ReptileTrainer(meta_model, inner_lr=0.01, meta_lr=0.001, device=device, reporter=reporter)

                # Create tasks for meta-learning (windowed chunks of training data)
                task_size = 200  # Size of each task window
                num_tasks = max(1, len(X_meta) // task_size - 1)

                tasks = []
                for i in range(num_tasks):
                    start_idx = i * task_size
                    end_idx = start_idx + task_size
                    task_X = X_meta[start_idx:end_idx]
                    task_y = y_meta[start_idx:end_idx]
                    tasks.append((task_X, task_y))

                # Train using Reptile
                if len(df_train_numeric) > 400:
                    reporter.set_status("Bake-Off: Meta-Learner", f"Starting Reptile meta-training...")

                    # Train for 10 epochs
                    trainer.train(df_train_numeric, feature_names, n_epochs=10, n_tasks_per_epoch=20, task_size=200)

                    # Wrap for backtester compatibility
                    model_params = {
                        'input_dim': input_dim,
                        'hidden_dim': 64,
                        'output_dim': 3,
                        'inner_lr': 0.01,
                        'meta_lr': 0.001,
                        'n_epochs': 10,
                        'n_tasks_per_epoch': 20,
                        'task_size': 200
                    }
                    wrapped_model = MetaLearnerWrapper(meta_model, feature_names, device, model_params)
                    explanation = "Meta-Learner: Neural network trained with Reptile meta-learning for rapid adaptation"

                    print(f"[Bake-Off] Meta-Learner training complete.")
                    reporter.log(f"[Bake-Off] Meta-Learner trained successfully")

                    return {"name": "Meta-Learner", "model": wrapped_model, "explanation": explanation, "dna": None}
                else:
                    print("[Bake-Off] Meta-Learner: Insufficient data for task creation.")
                    return None

            except Exception as e:
                print(f"[Bake-Off] ERROR: Meta-Learner training failed: {e}")
                logger.error(f"[Bake-Off] ERROR: Meta-Learner training failed: {e}", exc_info=True)
                return None

        # --- Run Bake-Off ---
        if market_environment == 'High_Volatility_Momentum':
            print("[Bake-Off] Prioritizing Momentum-based challengers.")
            gp_result = train_gp_islands(strategy_type='momentum')
            if gp_result:
                challengers.append(gp_result)
        elif market_environment == 'Low_Volatility_Reversion':
            print("[Bake-Off] Prioritizing Reversion-based challengers.")
            gp_result = train_gp_islands(strategy_type='reversion')
            if gp_result:
                challengers.append(gp_result)
            
            meta_result = train_meta_learner()
            if meta_result:
                challengers.append(meta_result)
        else: # Default case
            gp_result = train_gp_islands()
            if gp_result:
                challengers.append(gp_result)
            meta_result = train_meta_learner()
            if meta_result:
                challengers.append(meta_result)


        if not challengers:
            logger.warning("[Gauntlet] No challengers survived the Bake-Off.")
            return None, None
        
        # --- 5. The Gauntlet: Validate All Challengers ---
        print(f"\n{'-'*60}\n| FORGE: Entering Validation Gauntlet\n{'-'*60}")
        gauntlet_results = []
        for challenger in challengers:
            print(f"\n--- Validating Challenger: {challenger['name']} ---")
            reporter.set_status("Validation Gauntlet", f"Testing {challenger['name']}")
            # CRITICAL FIX: Pass actual training data for in-sample sanity check
            results = run_full_gauntlet(
                model=challenger['model'],
                df_processed=df_gauntlet_full.copy(),
                asset_symbol=asset_symbol,
                reporter=reporter,
                df_train_insample=df_train_full.copy(),  # Pass actual training data
                model_instance_id=model_instance_id # Pass the unique ID
            )
            gauntlet_results.append({**challenger, "results": results})

        # --- 6. Select the Champion ---
        print(f"\n{'='*60}\n| FORGE: Selecting Champion from Gauntlet Results\n{'='*60}")
        passed_challengers = [c for c in gauntlet_results if c["results"].get("all_passed", False)]

        if not passed_challengers:
            print("\n[FORGE] ❌ No models survived the gauntlet.")
            reporter.set_status("Finished", "No models passed the validation gauntlet.")
            return None, None

        champion_info = max(passed_challengers, key=lambda c: c["results"]["metrics"]["wfa_metrics"].get("sharpe_ratio", -1))
        
        # --- 7. Model Registration ---
        registry = ModelRegistry()
        model_id = registry.register_model(
            model_artifact=champion_info['model'],
            model_blueprint={"model_type": champion_info['name'], "explanation": champion_info['explanation']},
            validation_metrics=champion_info['results'].get("metrics", {}),
            asset_symbol=asset_symbol,
            xai_report_path=None,
            model_instance_id=model_instance_id # Pass the unique ID
        )
        
        winning_dna = champion_info.get("dna")
        return model_id, winning_dna

    except ImportError as e:
        error_msg = f"Failed to import necessary modules inside Forge cycle: {e}"
        reporter.set_status("Error", error_msg)
        if logger:
            logger.error(error_msg)
        print(error_msg)
        return None, None
    except Exception as e:
        error_msg = f"Forge cycle failed: {e}"
        logger.error(error_msg)
        return None, None

