import optuna
import numpy as np
import pandas as pd # Import pandas for isnull/isinf checks

from forge.crucible.backtester import VectorizedBacktester
from models_v2 import LGBMWrapper, XGBWrapper
from forge.models.transformer_wrapper import TransformerWrapper

# --- Define Optimization Constraints ---
from forge.evolution.opponent_templates import simple_rsi_opponent

# --- Define Optimization Constraints ---
MIN_TRADES = 30
MIN_PSR = 0.80 # 80% confidence that the strategy is statistically significant
MDD_THRESHOLD = 0.40 # Max acceptable drawdown
ADVERSARIAL_WEIGHT = 0.3 # How much to value counter-strategy performance

def run_fitness_evaluation(blueprint, X_train, y_train, X_val, y_val, returns_val, inferred_opponent_params=None):
    """
    The core fitness function for evaluating a single model blueprint.
    It trains the model, runs it through the high-fidelity backtester,
    and returns the metrics.
    Now includes "Dual Fitness" to evolve counter-strategies.
    """
    try:
        # --- 1. Model Training ---
        model_map = {"LGBMWrapper": LGBMWrapper, "XGBWrapper": XGBWrapper, "TransformerWrapper": TransformerWrapper}
        model_class = model_map.get(blueprint.architecture)
        if not model_class:
            raise ValueError(f"Unknown architecture: {blueprint.architecture}")

        valid_features = [f for f in blueprint.features if f in X_train.columns]
        if not valid_features:
            return VectorizedBacktester().get_default_metrics()

        model = model_class(**blueprint.hyperparameters)

        # --- DEFINITIVE FIX: Prepare data specifically for each model type ---
        X_train_sliced = X_train[valid_features]
        X_val_sliced = X_val[valid_features]

        if blueprint.architecture == "TransformerWrapper":
            # Transformer expects a single DataFrame with features and the label
            df_transformer_train = X_train_sliced.copy()
            df_transformer_train['label'] = y_train
            model.fit(df_transformer_train, device=blueprint.device)
            predictions = model.predict(X_val_sliced)
        else:
            # LGBM and XGB expect features and labels separately
            model.fit(X_train_sliced, y_train, device=blueprint.device)
            predictions = model.predict(X_val_sliced)

        # --- 2. Signal Generation ---
        entries = predictions == 1
        exits = predictions == -1

        signals = y_val.to_frame(name='label').copy()
        signals['entries'] = entries
        signals['exits'] = exits

        # --- 3. High-Fidelity Backtesting (Traditional Fitness) ---
        backtester = VectorizedBacktester()
        # --- FIX: Use a clean copy of the full X_val for backtesting ---
        backtest_data = X_val.copy()
        backtest_data['label'] = y_val

        metrics = backtester.run(backtest_data, signals)
        traditional_fitness = metrics.get("log_wealth", -1e9)

        # --- 4. Adversarial Fitness Calculation ---
        adversarial_fitness = 0
        if inferred_opponent_params is not None and len(inferred_opponent_params) > 0:
            try:
                # Generate opponent signals
                opponent_signals = simple_rsi_opponent(X_val, inferred_opponent_params)

                # Calculate PnL only when we trade against the opponent
                our_positions = pd.Series(0, index=X_val.index)
                our_positions[entries] = 1
                our_positions[exits] = 0 # We only care about entry timing for this calculation
                our_positions = our_positions.ffill().fillna(0)

                # Adversarial condition: we are long (1) and opponent is short (-1), or vice-versa
                adversarial_trades = (our_positions != 0) & (opponent_signals != 0) & (our_positions != opponent_signals)
                
                if adversarial_trades.any():
                    # Calculate returns for the periods we are in an adversarial position
                    adversarial_returns = returns_val[adversarial_trades] * our_positions[adversarial_trades]
                    # Adversarial fitness is the cumulative log return during these periods
                    adversarial_fitness = np.log1p(adversarial_returns).sum()
                    
                metrics["adversarial_fitness"] = adversarial_fitness
            except Exception as e:
                print(f"[Fitness] Error during adversarial calculation: {e}")
                metrics["adversarial_fitness"] = 0

        # --- 5. Dual Fitness Combination ---
        final_fitness = ((1 - ADVERSARIAL_WEIGHT) * traditional_fitness) + (ADVERSARIAL_WEIGHT * adversarial_fitness)
        metrics["log_wealth"] = final_fitness # Overwrite with the combined score

        return metrics

    except Exception as e:
        print(f"[Fitness] Error during evaluation: {e}")
        return VectorizedBacktester().get_default_metrics()

# --- objective function remains unchanged ---
def objective(trial: optuna.Trial, ga_instance) -> float:
    """
    The main objective function for Optuna, designed to be called by the Genetic Algorithm.
    """
    # Let the GA instance handle the creation of the blueprint from the trial
    blueprint = ga_instance.create_blueprint_from_trial(trial)

    # --- ACP INTEGRATION: Get inferred opponent params from the GA instance ---
    inferred_opponent_params = getattr(ga_instance, 'inferred_opponent_params', None)

    # --- Run the full evaluation ---
    metrics = run_fitness_evaluation(
        blueprint,
        ga_instance.X_train, ga_instance.y_train,
        ga_instance.X_val, ga_instance.y_val,
        ga_instance.returns_val,
        inferred_opponent_params=inferred_opponent_params # Pass them to the fitness function
    )

    # --- Constraint Handling (The Brakes) ---
    trial.set_user_attr("total_trades", metrics["total_trades"])
    trial.set_user_attr("psr", metrics["probabilistic_sharpe_ratio"])
    trial.set_user_attr("max_drawdown", metrics["max_drawdown"])
    trial.set_user_attr("log_wealth", metrics["log_wealth"]) # This is now the DUAL fitness
    trial.set_user_attr("adversarial_fitness", metrics.get("adversarial_fitness", 0))


    if metrics["total_trades"] < MIN_TRADES:
        raise optuna.TrialPruned(f"Not enough trades: {metrics['total_trades']}")

    if metrics["probabilistic_sharpe_ratio"] < MIN_PSR:
        raise optuna.TrialPruned(f"PSR too low: {metrics['probabilistic_sharpe_ratio']:.2f}")

    if metrics["max_drawdown"] > MDD_THRESHOLD:
        raise optuna.TrialPruned(f"MDD too high: {metrics['max_drawdown']:.2f}")

    # --- The Engine (Maximize Growth Velocity) ---
    final_fitness = metrics["log_wealth"]

    # Ensure finite value before returning
    return final_fitness if np.isfinite(final_fitness) else -1e9