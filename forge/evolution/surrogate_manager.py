# In forge/evolution/surrogate_manager.py

import numpy as np
import pandas as pd
import joblib
import time
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer # Added for NaN handling
from multiprocessing import Pool, cpu_count
import multiprocessing
import os

# --- Assuming run_fitness_evaluation is correctly importable ---
# Adjust the import path if necessary based on your project structure
try:
    # This path assumes surrogate_manager.py is in forge/evolution/
    from forge.crucible.fitness_function import run_fitness_evaluation
    FITNESS_FUNC = run_fitness_evaluation
    print("[SurrogateManager] Successfully imported run_fitness_evaluation") # DEBUG
except ImportError:
    print("[SurrogateManager] ERROR: Could not import run_fitness_evaluation!") # DEBUG
    # Define a dummy function if import fails, to prevent crashes but highlight the issue
    def dummy_fitness_func(*args, **kwargs):
        print("[SurrogateManager] ERROR: DUMMY FITNESS FUNCTION CALLED!") # DEBUG
        return {"log_wealth": -1e9, "total_trades": 0, "psr": 0, "max_drawdown": 1.0}
    FITNESS_FUNC = dummy_fitness_func

# --- Worker Data for Multiprocessing ---
worker_data = {}

def initialize_worker_surrogate(X_train, y_train, X_val, y_val, returns_val, fitness_func):
    """Initializer for Pool workers used by SurrogateManager."""
    worker_data['X_train'] = X_train
    worker_data['y_train'] = y_train
    worker_data['X_val'] = X_val
    worker_data['y_val'] = y_val
    worker_data['returns_val'] = returns_val
    worker_data['fitness_func'] = fitness_func
    print(f"[Worker {os.getpid()}] Initialized with fitness func: {fitness_func.__name__}") # DEBUG

def evaluate_blueprint_actual_worker(blueprint):
    """Worker function to evaluate a single blueprint using the actual fitness function."""
    # --- DEBUG: Print worker PID ---
    # print(f"[Worker {os.getpid()}] Starting evaluation for: {blueprint}")
    # ---
    try:
        X_train = worker_data.get('X_train')
        y_train = worker_data.get('y_train')
        X_val = worker_data.get('X_val')
        y_val = worker_data.get('y_val')
        returns_val = worker_data.get('returns_val')
        fitness_func = worker_data.get('fitness_func')

        if X_train is None or X_val is None or fitness_func is None:
            print(f"[Worker {os.getpid()}] ERROR: Missing data or fitness_func in worker!") # DEBUG
            return -1e9

        # --- DEBUG: Confirming call ---
        # print(f"[Worker {os.getpid()}] Calling actual fitness function: {fitness_func.__name__}")
        # ---
        
        metrics = fitness_func(blueprint, X_train, y_train, X_val, y_val, returns_val)
        fitness = metrics.get("log_wealth", -1e9)

        # --- DEBUG: Print result ---
        # print(f"[Worker {os.getpid()}] Evaluation complete. Fitness: {fitness}")
        # ---
        return fitness if np.isfinite(fitness) else -1e9
    except Exception as e:
        print(f"[Worker {os.getpid()}] EXCEPTION during actual evaluation for {blueprint}: {e}") # DEBUG
        # import traceback
        # print(traceback.format_exc()) # Uncomment for full traceback
        return -1e9


class SurrogateManager:
    """Manages the surrogate model for fitness prediction."""
    def __init__(self, model_path="models/fitness_oracle.pkl", update_interval=5, min_samples_update=50,
                 X_train=None, y_train=None, X_val=None, y_val=None, returns_val=None,
                 num_processes=None, logger=None):
        self.model_path = model_path
        self.update_interval = update_interval # Generations between retraining
        self.min_samples_update = min_samples_update # Min new samples needed
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean') # Handle NaNs
        self.data_store = pd.DataFrame() # Store features and actual fitness
        self.generation_counter = 0
        self.num_processes = num_processes if num_processes is not None else max(1, cpu_count() // 2) # Use fewer cores for surrogate tasks potentially
        self.logger = logger if logger else logging.getLogger(__name__)

        # Store data needed for actual evaluations
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.returns_val = returns_val

        self.feature_names_ = None # Store feature names used by the trained model

        self._load_model()

    def _prepare_features(self, blueprints):
        """Converts blueprints into a feature matrix suitable for the model."""
        data = []
        for bp in blueprints:
            features = {}
            # --- Numerical Features ---
            # Architecture (one-hot encode?) - Simple ordinal encoding for now
            arch_map = {"LGBMWrapper": 0, "XGBWrapper": 1, "TransformerWrapper": 2, "BayesianNN": 3}
            features['arch'] = arch_map.get(bp.architecture, -1)
            features['horizon'] = bp.training_horizon
            features['sl'] = bp.risk_parameters.get('stop_loss_multiplier', 0)
            features['tp'] = bp.risk_parameters.get('take_profit_multiplier', 0)

            # Hyperparameters (flatten, handle missing) - Use defaults or mean/median?
            # Example for LGBM - expand this for all models/params
            features['hp_n_estimators'] = bp.hyperparameters.get('n_estimators', 100) # Use default/mean?
            features['hp_num_leaves'] = bp.hyperparameters.get('num_leaves', 31)
            features['hp_max_depth'] = bp.hyperparameters.get('max_depth', -1) # XGB specific?
            # ... add all hyperparameters defined in genetic_algorithm.py ...
            features['hp_model_dim'] = bp.hyperparameters.get('model_dim', 32)
            features['hp_nhead'] = bp.hyperparameters.get('nhead', 4)
            features['hp_num_layers'] = bp.hyperparameters.get('num_layers', 2)
            features['hp_epochs'] = bp.hyperparameters.get('epochs', 10)
            features['hp_sequence_length'] = bp.hyperparameters.get('sequence_length', 20)


            # --- Categorical/Set Features (Tricky) ---
            # Feature presence (binary/count?)
            # Use multi-hot encoding based on a fixed list of ALL possible features
            # This requires knowing all features defined in genetic_algorithm.py
            # Let's get this list dynamically if possible, or define it statically
            all_defined_features = self._get_all_possible_features()
            for feat_name in all_defined_features:
                 features[f'uses_{feat_name}'] = 1 if feat_name in bp.features else 0

            data.append(features)

        df = pd.DataFrame(data)
        # Ensure columns match what the model was trained on, handle missing/new cols
        if self.feature_names_:
             # Add missing columns with default value (0 or mean from imputer)
             for col in self.feature_names_:
                 if col not in df.columns:
                     df[col] = 0 # Or use self.imputer.statistics_ if fitted?
             # Reorder and select columns to match training
             df = df[self.feature_names_]

        return df

    def _get_all_possible_features(self):
        """Helper to get all feature names defined in the GA."""
        # This might need adjustment based on how features are defined
        try:
            from forge.blueprint_factory.genetic_algorithm import FEATURE_SUBSETS
            all_feats = set()
            for group_name, feat_list in FEATURE_SUBSETS.items():
                all_feats.update(feat_list)
            return sorted(list(all_feats))
        except ImportError:
            print("[SurrogateManager] WARNING: Could not import FEATURE_SUBSETS. Feature encoding might be incomplete.")
            # Fallback: return features seen so far? Risky. Return empty for now.
            return []
        except Exception as e:
            print(f"[SurrogateManager] ERROR getting features: {e}")
            return []


    def _load_model(self):
        """Loads the surrogate model and scaler from disk."""
        try:
            load_data = joblib.load(self.model_path)
            self.model = load_data['model']
            self.scaler = load_data['scaler']
            self.imputer = load_data['imputer']
            self.data_store = load_data.get('data_store', pd.DataFrame()) # Load history
            self.feature_names_ = load_data.get('feature_names') # Load feature names
            self.logger.info(f"Oracle: Model loaded from {self.model_path} ({len(self.data_store)} samples)")
            if self.feature_names_ is None:
                self.logger.warning("Oracle: Loaded model has no feature names stored.")
            # Ensure data_store columns align if they exist
            if not self.data_store.empty and self.feature_names_ and 'actual_fitness' in self.data_store.columns:
                 expected_cols = self.feature_names_ + ['actual_fitness']
                 self.data_store = self.data_store.reindex(columns=expected_cols) # Align columns


        except FileNotFoundError:
            self.logger.warning(f"Oracle: No model found at {self.model_path}. Will train from scratch.")
            self.model = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42, max_depth=10, min_samples_leaf=5) # Example model
        except Exception as e:
            self.logger.error(f"Oracle: Error loading model: {e}. Will train from scratch.")
            self.model = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42, max_depth=10, min_samples_leaf=5)


    def _save_model(self):
        """Saves the surrogate model, scaler, imputer, history, and feature names."""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            save_data = {
                'model': self.model,
                'scaler': self.scaler,
                'imputer': self.imputer,
                'data_store': self.data_store,
                'feature_names': self.feature_names_ # Save feature names
            }
            joblib.dump(save_data, self.model_path)
            self.logger.info(f"Oracle: Model saved to {self.model_path}")
        except Exception as e:
            self.logger.error(f"Oracle: Error saving model: {e}")

    def update_datastore(self, evaluated_blueprints):
        """Adds newly evaluated blueprints to the historical data."""
        if not evaluated_blueprints:
            return

        new_data_list = []
        for bp in evaluated_blueprints:
            if bp.fitness != -1.0 and np.isfinite(bp.fitness): # Only add valid, evaluated blueprints
                features = self._prepare_features([bp]).iloc[0].to_dict() # Get features for this blueprint
                features['actual_fitness'] = bp.fitness
                new_data_list.append(features)

        if not new_data_list:
            return

        new_data = pd.DataFrame(new_data_list)

        # Align columns before concatenating
        if not self.data_store.empty:
            # Ensure new_data has all columns from data_store, fill missing with NaN
            for col in self.data_store.columns:
                if col not in new_data.columns:
                    new_data[col] = np.nan
            new_data = new_data[self.data_store.columns] # Match order and selection

            self.data_store = pd.concat([self.data_store, new_data], ignore_index=True)
        else:
             # First time adding data
             self.data_store = new_data
             # Set feature names based on this first batch
             self.feature_names_ = [col for col in self.data_store.columns if col != 'actual_fitness']


        # Optional: Limit datastore size?
        # self.data_store = self.data_store.tail(MAX_DATASTORE_SIZE)

        self.logger.info(f"Oracle: Added {len(new_data)} new samples. Datastore size: {len(self.data_store)}")


    def train(self):
        """Trains/retrains the surrogate model on the stored data."""
        if len(self.data_store) < self.min_samples_update: # Need enough data
            self.logger.info(f"Oracle: Skipping training, not enough samples ({len(self.data_store)}/{self.min_samples_update})")
            return

        self.logger.info(f"Oracle: Training on {len(self.data_store)} samples...")
        start_time = time.time()

        # Separate features (X) and target (y)
        X = self.data_store.drop('actual_fitness', axis=1)
        y = self.data_store['actual_fitness']

        # Ensure feature names are stored based on this training data
        self.feature_names_ = list(X.columns)

        # Impute NaNs (before scaling)
        X_imputed = self.imputer.fit_transform(X)
        X_imputed_df = pd.DataFrame(X_imputed, columns=self.feature_names_) # Keep as DataFrame

        # Scale features
        X_scaled = self.scaler.fit_transform(X_imputed_df)

        # Handle potential NaNs in target 'y' (e.g., if loading old data)
        valid_indices = y.notna() & np.isfinite(y)
        if not valid_indices.all():
             self.logger.warning(f"Oracle: Found {sum(~valid_indices)} NaN/inf values in target variable. Removing them before training.")
             X_scaled = X_scaled[valid_indices]
             y = y[valid_indices]

        if len(y) < self.min_samples_update:
             self.logger.error("Oracle: Not enough valid samples remain after cleaning target variable. Aborting training.")
             return


        # Train the model
        try:
             self.model.fit(X_scaled, y)
             end_time = time.time()
             self.logger.info(f"Oracle: Training complete. Took {end_time - start_time:.2f}s")

             # Evaluate (optional, on a hold-out set or via cross-val)
             # For simplicity, evaluate on the training set (biased)
             y_pred = self.model.predict(X_scaled)
             mae = mean_absolute_error(y, y_pred)
             r2 = r2_score(y, y_pred)
             self.logger.info(f"Oracle: Training MAE: {mae:.4f}, R²: {r2:.4f}")

             self._save_model() # Save after successful training
        except Exception as e:
             self.logger.error(f"Oracle: Error during model training: {e}")


    def predict(self, blueprints):
        """Predicts fitness for a list of blueprints using the surrogate."""
        if self.model is None or not self.feature_names_:
            self.logger.warning("Oracle: Model not trained or feature names missing. Returning default score.")
            return np.full(len(blueprints), -1e9) # Return default low score

        if not blueprints:
            return np.array([])

        try:
            # Prepare features, ensuring column order matches training data
            X = self._prepare_features(blueprints)

            # Impute and Scale
            X_imputed = self.imputer.transform(X) # Use transform, not fit_transform
            X_scaled = self.scaler.transform(X_imputed) # Use transform

            predictions = self.model.predict(X_scaled)
            return predictions
        except Exception as e:
             self.logger.error(f"Oracle: Error during prediction: {e}. Returning default scores.")
             # import traceback
             # print(traceback.format_exc()) # Uncomment for details
             return np.full(len(blueprints), -1e9)


    def screen_and_evaluate(self, population, generation):
        """
        Screens population with surrogate, evaluates top candidates,
        updates datastore, and retrains surrogate periodically.
        """
        self.generation_counter = generation # Keep track of generation

        # 1. Predict fitness for all individuals using the surrogate
        self.logger.info(f"SAE: Oracle screening {len(population)} individuals...")
        predicted_fitness = self.predict(population)

        # Assign predicted fitness temporarily (e.g., to a temp attribute)
        for bp, pred_fit in zip(population, predicted_fitness):
            bp.predicted_fitness = pred_fit # Use a temporary attribute

        # 2. Select top N% candidates based on predicted fitness
        # Sort by predicted fitness descending
        population.sort(key=lambda bp: getattr(bp, 'predicted_fitness', -1e9), reverse=True)
        
        # --- DEBUG ---
        # best_pred = getattr(population[0], 'predicted_fitness', 'N/A') if population else 'N/A'
        # print(f"DEBUG: Best predicted fitness: {best_pred}")
        # ---

        # Determine number to evaluate (e.g., top 2% or min 10)
        num_to_evaluate = max(10, int(len(population) * 0.02))
        candidates = population[:num_to_evaluate]

        self.logger.info(f"SAE: Oracle selected top {len(candidates)} candidates ({len(candidates)/len(population)*100:.1f}%)")

        # 3. Evaluate selected candidates with the actual fitness function
        evaluated_candidates = self.evaluate_actual(candidates) # This returns candidates with bp.fitness updated

        # 4. Update the historical data store with newly evaluated candidates
        self.update_datastore(evaluated_candidates)

        # 5. Assign fitness: Use actual for evaluated, predicted for others? Or just actual?
        # For simplicity in GA, let's assign actual fitness where available,
        # and potentially a default low score or predicted score for others.
        # It's crucial the GA gets *some* fitness value. Assigning actual fitness
        # to the evaluated ones is the most direct approach. Others keep their old fitness or -1.0.
        # (The GA loop in genetic_algorithm.py should handle fitness assignment)

        # 6. Retrain surrogate model periodically
        if self.generation_counter % self.update_interval == 0:
            self.train()

        # Return the candidates that were actually evaluated (for logging/HoF update potentially)
        # The GA itself will use the .fitness attribute set on the blueprints in the main population list.
        return evaluated_candidates


    def evaluate_actual(self, blueprints):
        """Evaluates a list of blueprints using the *actual* fitness function, possibly in parallel."""
        if not blueprints:
            return []

        # --- DEBUG PRINTS ---
        print(f"\n--- DEBUG: Entering SurrogateManager.evaluate_actual for {len(blueprints)} blueprints ---")
        print(f"DEBUG: First blueprint to evaluate: {blueprints[0]}")
        print(f"DEBUG: Using fitness function: {FITNESS_FUNC.__name__}")
        print(f"DEBUG: Multiprocessing enabled: {self.num_processes > 1}")
        # ---

        start_time = time.time()
        self.logger.info(f"SAE: Evaluating {len(blueprints)} individuals with actual backtests...")

        # Prepare data required by the fitness function
        if self.X_train is None or self.X_val is None:
             self.logger.error("SAE: Missing X_train or X_val data for actual evaluation!")
             print("DEBUG: ERROR - Missing X_train or X_val in evaluate_actual!") # DEBUG
             # Assign error score to all blueprints and return
             for bp in blueprints: bp.fitness = -1e9
             return blueprints

        # Decide whether to run sequentially or in parallel
        is_subprocess = multiprocessing.current_process().name != 'MainProcess'
        if is_subprocess or self.num_processes <= 1:
            # Sequential execution
            print("DEBUG: Running actual evaluation sequentially...") # DEBUG
            actual_fitness_values = []
            for i, bp in enumerate(blueprints):
                 # print(f"DEBUG: Evaluating blueprint {i+1}/{len(blueprints)} sequentially...") # DEBUG - Can be noisy
                 try:
                      metrics = FITNESS_FUNC(bp, self.X_train, self.y_train, self.X_val, self.y_val, self.returns_val)
                      fitness = metrics.get("log_wealth", -1e9)
                      actual_fitness_values.append(fitness if np.isfinite(fitness) else -1e9)
                 except Exception as e:
                      print(f"DEBUG: EXCEPTION in sequential evaluation for {bp}: {e}") # DEBUG
                      actual_fitness_values.append(-1e9)
        else:
            # Parallel execution using Pool
            print(f"DEBUG: Running actual evaluation in parallel with {self.num_processes} workers...") # DEBUG
            initializer_args = (
                self.X_train, self.y_train, self.X_val, self.y_val, self.returns_val, FITNESS_FUNC
            )
            try:
                with Pool(processes=self.num_processes, initializer=initialize_worker_surrogate, initargs=initializer_args) as pool:
                    # Use imap for potentially better memory usage and responsiveness
                    results_iterator = pool.imap(evaluate_blueprint_actual_worker, blueprints)
                    actual_fitness_values = list(results_iterator) # Collect results
                print(f"DEBUG: Parallel evaluation finished. Got {len(actual_fitness_values)} results.") # DEBUG
            except Exception as e:
                print(f"DEBUG: EXCEPTION during parallel pool execution: {e}") # DEBUG
                # Fallback or error handling
                actual_fitness_values = [-1e9] * len(blueprints) # Assign error score on pool failure


        # Assign the calculated actual fitness back to the blueprints
        if len(actual_fitness_values) == len(blueprints):
            for bp, fitness in zip(blueprints, actual_fitness_values):
                bp.fitness = fitness
            print(f"DEBUG: Assigned actual fitness to {len(blueprints)} blueprints.") # DEBUG
        else:
             print(f"DEBUG: ERROR - Mismatch in blueprint count ({len(blueprints)}) and fitness results ({len(actual_fitness_values)})!") # DEBUG
             # Assign error score if counts mismatch
             for bp in blueprints: bp.fitness = -1e9


        end_time = time.time()
        self.logger.info(f"SAE: Evaluation complete. {len(blueprints)} individuals evaluated. Took {end_time - start_time:.2f}s")
        print(f"--- DEBUG: Exiting SurrogateManager.evaluate_actual ---") # DEBUG

        return blueprints # Return the list with updated .fitness attributes