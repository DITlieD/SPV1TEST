import optuna
import numpy as np
import pandas as pd # Import pandas for isnull/isinf checks

from forge.crucible.backtester import VectorizedBacktester
from models_v2 import LGBMWrapper, XGBWrapper
from forge.models.transformer_wrapper import TransformerWrapper

# --- Define Optimization Constraints ---
MIN_TRADES = 30
MIN_PSR = 0.80 # 80% confidence that the strategy is statistically significant
MDD_THRESHOLD = 0.40 # Max acceptable drawdown

def run_fitness_evaluation(blueprint, X_train, y_train, X_val, y_val, returns_val):
    """
    The core fitness function for evaluating a single model blueprint.
    It trains the model, runs it through the high-fidelity backtester,
    and returns the metrics.
    """
    # --- START DEBUG PRINTS ---
    print("\n--- DEBUG: Entering run_fitness_evaluation ---")
    print(f"Blueprint: {blueprint}")
    if X_train is None or X_val is None or y_train is None or y_val is None:
        print("DEBUG: X_train, X_val, y_train, or y_val is None!")
        return VectorizedBacktester().get_default_metrics()

    print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
    # print(f"X_train columns: {list(X_train.columns)}") # Uncomment if needed, can be long
    # print(f"X_val columns: {list(X_val.columns)}") # Uncomment if needed, can be long
    print(f"y_train unique: {np.unique(y_train)}, y_val unique: {np.unique(y_val)}")
    # --- END DEBUG PRINTS ---

    try:
        # --- 1. Model Training ---
        model_map = {"LGBMWrapper": LGBMWrapper, "XGBWrapper": XGBWrapper, "TransformerWrapper": TransformerWrapper}
        model_class = model_map.get(blueprint.architecture)
        if not model_class:
            print(f"DEBUG: Unknown architecture: {blueprint.architecture}") # DEBUG
            raise ValueError(f"Unknown architecture: {blueprint.architecture}")

        # --- DEBUG: Print blueprint features ---
        print(f"DEBUG: Blueprint features requested: {list(blueprint.features)}")
        # ---

        # Use only the features specified in the blueprint that exist in the training data
        valid_features = [f for f in blueprint.features if f in X_train.columns]

        # --- DEBUG: Print valid features ---
        print(f"DEBUG: Valid features found in X_train: {valid_features}")
        # ---

        # --- FIX: Check for valid features *before* training ---
        if not valid_features:
            print(f"[Fitness] Error: No valid features found for blueprint. Blueprint wanted: {list(blueprint.features)}. Data has: {list(X_train.columns)}")
            return VectorizedBacktester().get_default_metrics()

        # --- DEBUG: Check for NaNs/Infs before training ---
        X_train_slice = X_train[valid_features]
        if X_train_slice.isnull().values.any() or np.isinf(X_train_slice.values).any():
            print(f"DEBUG: NaNs or Infs found in X_train slice for training!")
            # Optional: print specific columns with issues
            # print(X_train_slice.isnull().sum())
            # print(np.isinf(X_train_slice).sum())
            return VectorizedBacktester().get_default_metrics() # Treat as failure
        # ---

        model = model_class(**blueprint.hyperparameters)
        print(f"DEBUG: Training model {blueprint.architecture}...") # DEBUG
        
        if blueprint.architecture == "TransformerWrapper":
            # TransformerWrapper expects a single DataFrame with features and labels
            df_transformer_train = X_train_slice.copy()
            df_transformer_train['label'] = y_train
            model.fit(df_transformer_train, device=blueprint.device) # Pass device explicitly
        else:
            model.fit(X_train_slice, y_train, device=blueprint.device) # Pass device explicitly
        print("DEBUG: Model training complete.") # DEBUG

        # --- 2. Signal Generation ---
        # --- DEBUG: Check for NaNs/Infs before prediction ---
        X_val_slice = X_val[valid_features]
        if X_val_slice.isnull().values.any() or np.isinf(X_val_slice.values).any():
            print(f"DEBUG: NaNs or Infs found in X_val slice for prediction!")
            return VectorizedBacktester().get_default_metrics() # Treat as failure
        # ---
        
        print("DEBUG: Generating predictions...") # DEBUG
        predictions = model.predict(X_val_slice)
        print(f"DEBUG: Predictions generated. Unique values: {np.unique(predictions)}") # DEBUG

        # --- THIS IS THE FIX ---
        # Signal logic must match the labels from get_specialist_labels (1 for long, -1 for short)
        entries = predictions == 1
        exits = predictions == -1 # Changed from 'predictions == 2'
        # --- END OF FIX ---

        # --- DEBUG: Check signal counts ---
        print(f"DEBUG: Entry signals: {entries.sum()}, Exit signals: {exits.sum()}")
        # ---

        signals = y_val.to_frame(name='label').copy()
        signals['entries'] = entries
        signals['exits'] = exits

        # --- 3. High-Fidelity Backtesting ---
        backtester = VectorizedBacktester()
        
        # --- DEBUG: Check required columns before backtest ---
        required_cols = ['open', 'close']
        atr_col = next((c for c in X_val.columns if c.startswith('ATRr')), None)
        if atr_col:
            required_cols.append(atr_col)
        else:
             print("[Fitness] Error: No 'ATRr' column (e.g., 'ATRr_14') in X_val. Cannot run backtest.")
             return VectorizedBacktester().get_default_metrics() # Exit before backtest

        if not all(col in X_val.columns for col in required_cols):
             print(f"[Fitness] Error: Required backtest columns missing. Needed: {required_cols}. Have: {list(X_val.columns)}")
             return VectorizedBacktester().get_default_metrics() # Exit before backtest
        # ---

        backtest_data = X_val.copy() # Use full X_val for backtester context
        backtest_data['close'] = y_val # Backtester uses 'close' from this merge
        backtest_data['label'] = y_val # Add the 'label' column to backtest_data

        print("DEBUG: Starting backtester...") # DEBUG
        metrics = backtester.run(backtest_data, signals)
        print(f"DEBUG: Backtester finished. Log Wealth: {metrics.get('log_wealth', 'N/A')}") # DEBUG
        
        # --- DEBUG: Check if default metrics were returned by backtester ---
        if metrics.get("log_wealth", -1e9) == -1e9 and metrics.get("total_trades", -1) == 0:
            print("DEBUG: Backtester returned default metrics (likely zero trades or error).")
        # ---
        
        return metrics

    except Exception as e:
        print(f"[Fitness] Error during evaluation: {e}") # Make errors visible
        # Optional: Print traceback for detailed debugging
        # import traceback
        # print(traceback.format_exc())
        return VectorizedBacktester().get_default_metrics()

# --- objective function remains unchanged ---
def objective(trial: optuna.Trial, ga_instance) -> float:
    """
    The main objective function for Optuna, designed to be called by the Genetic Algorithm.
    """
    # Let the GA instance handle the creation of the blueprint from the trial
    blueprint = ga_instance.create_blueprint_from_trial(trial)

    # --- Run the full evaluation ---
    metrics = run_fitness_evaluation(
        blueprint,
        ga_instance.X_train, ga_instance.y_train,
        ga_instance.X_val, ga_instance.y_val,
        ga_instance.returns_val
    )

    # --- Constraint Handling (The Brakes) ---
    trial.set_user_attr("total_trades", metrics["total_trades"])
    trial.set_user_attr("psr", metrics["probabilistic_sharpe_ratio"])
    trial.set_user_attr("max_drawdown", metrics["max_drawdown"])
    trial.set_user_attr("log_wealth", metrics["log_wealth"])

    if metrics["total_trades"] < MIN_TRADES:
        raise optuna.TrialPruned(f"Not enough trades: {metrics['total_trades']}")

    if metrics["probabilistic_sharpe_ratio"] < MIN_PSR:
        raise optuna.TrialPruned(f"PSR too low: {metrics['probabilistic_sharpe_ratio']:.2f}")

    if metrics["max_drawdown"] > MDD_THRESHOLD:
        raise optuna.TrialPruned(f"MDD too high: {metrics['max_drawdown']:.2f}")

    # --- The Engine (Maximize Growth Velocity) ---
    log_wealth = metrics["log_wealth"]

    # Ensure finite value before returning
    return log_wealth if np.isfinite(log_wealth) else -1e9