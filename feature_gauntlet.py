# feature_gauntlet_fixed.py
"""
NOTE: This script is a standalone utility for automated feature discovery using
Genetic Programming. It is not part of the primary, live adaptive loop, which
is orchestrated by 'forge/overlord/task_scheduler.py' (the Forge).
"""
import os
import pandas as pd
import numpy as np
from gplearn.genetic import SymbolicRegressor
import re

from data_processing_v2 import get_market_data, add_all_features, get_specialist_labels
from models_v2 import SimpleConfluenceModel
from backtester import Backtester
from forge.crucible.fitness_function import calculate_v3_fitness

# --- Guardrail Configuration ---
SIMPLICITY_THRESHOLD = 15
CORRELATION_THRESHOLD = 0.9
PERFORMANCE_UPLIFT_MARGIN = 0.1

def run_feature_synthesis(X, y):
    """
    Generates a pool of candidate features using Genetic Programming.
    """
    print("\n[Genesis] --- Step 1: Generating Candidate Features with GP ---")
    est_gp = SymbolicRegressor(population_size=1000, generations=10,
                               stopping_criteria=0.01, p_crossover=0.7,
                               p_subtree_mutation=0.1, p_hoist_mutation=0.05,
                               p_point_mutation=0.1, max_samples=0.9, verbose=1,
                               parsimony_coefficient=0.01, random_state=0,
                               function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min'))
    
    est_gp.fit(X, y)
    print("[Genesis] --- GP Feature Generation Complete ---")
    return [est_gp._program]

def apply_guardrails(candidates, df_features):
    """
    Applies a series of guardrails to vet the candidate features.
    """
    print("\n[Genesis] --- Step 2: Applying Guardrails to Candidates ---")
    approved_features = []
    
    for i, candidate in enumerate(candidates):
        print(f"\n[Genesis] --- Vetting Candidate {i+1}/{len(candidates)}: {candidate} ---")
        
        if len(str(candidate)) > SIMPLICITY_THRESHOLD:
            print("  -> ❌ FAILED: Exceeds simplicity threshold.")
            continue
        print("  -> ✅ PASSED: Simplicity check.")

        try:
            def protected_division(x1, x2):
                with np.errstate(divide='ignore', invalid='ignore'):
                    return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)
            def protected_sqrt(x1): return np.sqrt(np.abs(x1))
            def protected_log(x1):
                with np.errstate(divide='ignore', invalid='ignore'):
                    return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)
            def protected_inverse(x1):
                with np.errstate(divide='ignore', invalid='ignore'):
                    return np.where(np.abs(x1) > 0.001, 1. / x1, 0.)

            feature_map = {f'X{i}': df_features[col] for i, col in enumerate(df_features.columns)}
            eval_globals = {
                'add': np.add, 'sub': np.subtract, 'mul': np.multiply, 'div': protected_division,
                'sqrt': protected_sqrt, 'log': protected_log, 'abs': np.abs, 'neg': np.negative,
                'inv': protected_inverse, 'max': np.maximum, 'min': np.minimum,
            }
            
            candidate_series = eval(str(candidate), eval_globals, feature_map)

            if candidate_series.isnull().all():
                print("  -> ❌ FAILED: Feature produces all NaNs.")
                continue
                
            correlation = candidate_series.corr(df_features['close'])
            if abs(correlation) > CORRELATION_THRESHOLD:
                print(f"  -> ❌ FAILED: High correlation ({correlation:.2f}).")
                continue
            print("  -> ✅ PASSED: Redundancy check.")
        except Exception as e:
            print(f"  -> ❌ FAILED: Could not evaluate feature: {e}")
            continue

        df_test = df_features.copy()
        df_test['candidate'] = candidate_series
        df_base = df_test.drop(columns=['candidate'])
        df_challenger = df_test.copy()

        model_base = SimpleConfluenceModel()
        model_base.fit(df_base.iloc[:-1000])
        predictions_base = model_base.predict(df_base.iloc[-1000:])
        
        model_challenger = SimpleConfluenceModel()
        model_challenger.fit(df_challenger.iloc[:-1000])
        predictions_challenger = model_challenger.predict(df_challenger.iloc[-1000:])

        true_returns = df_test.iloc[-1000:]['close'].pct_change()
        fitness_base = calculate_v3_fitness(true_returns, predictions_base)
        fitness_challenger = calculate_v3_fitness(true_returns, predictions_challenger)

        if fitness_challenger > fitness_base * (1 + PERFORMANCE_UPLIFT_MARGIN):
            print(f"  -> ✅ PASSED: Velocity uplift confirmed (Fitness: {fitness_base:.4f} -> {fitness_challenger:.4f}).")
            approved_features.append(candidate)
        else:
            print(f"  -> ❌ FAILED: No significant velocity uplift (Fitness: {fitness_base:.4f} -> {fitness_challenger:.4f}).")
            
    return approved_features

import json

def integrate_features(approved_features):
    """Saves the string representations of approved feature formulas to the GP Foundry."""
    print(f"\n[Genesis] --- Step 3: Storing {len(approved_features)} Approved Features in the Foundry ---")

    # Convert gplearn's internal program objects to strings
    feature_formulas = [str(feature) for feature in approved_features]

    # This path should be accessible by the main application
    output_path = 'forge/data_processing/gp_feature_foundry.json'
    try:
        # Read existing features to avoid duplicates and append new ones
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                existing_formulas = json.load(f)
        else:
            existing_formulas = []

        # Add only new, unique formulas
        new_unique_formulas = [f for f in feature_formulas if f not in existing_formulas]
        if not new_unique_formulas:
            print("  -> No new unique features to add. Foundry is up to date.")
            return

        all_formulas = existing_formulas + new_unique_formulas
        with open(output_path, 'w') as f:
            json.dump(all_formulas, f, indent=4)
        print(f"  -> ✅ Success: Added {len(new_unique_formulas)} new features. Foundry now contains {len(all_formulas)} total features.")

    except Exception as e:
        print(f"  -> ❌ FAILED: Could not save features to Foundry. Error: {e}")

if __name__ == '__main__':
    asset = 'BTC/USDT'
    df_full = get_market_data(asset, '4h', limit=10000)
    df_features = add_all_features(df_full.copy())
    atr_col = next((c for c in df_features.columns if c.startswith('ATRr')), 'ATRr_14')
    df_features['label'] = get_specialist_labels(df_features, atr_col, module_type='reversion')
    df_features.dropna(inplace=True)
    
    X = df_features.drop(columns=['label'])
    y = df_features['label']
    
    candidate_features = run_feature_synthesis(X, y)
    approved_features = apply_guardrails(candidate_features, df_features)
    
    if approved_features:
        integrate_features(approved_features)
        print("\n[Genesis] --- Step 4: Final Sanity Check ---")
        try:
            from chimera_trainer import train_initial_model
            train_initial_model('DOGE/USDT', '4h', 'models')
            print("  -> ✅ PASSED: System can still train a model with the new features.")
        except Exception as e:
            print(f"  -> ❌ FAILED: An error occurred during the sanity check: {e}")
    else:
        print("\n[Genesis] --- No new features were approved. ---")
