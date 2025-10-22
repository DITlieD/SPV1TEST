# forge/evolution/feature_synthesizer.py

import logging
import pandas as pd
import numpy as np
# Use SymbolicTransformer which is specifically designed for feature generation
from gplearn.genetic import SymbolicTransformer
import warnings
import sys
import os
import multiprocessing # Import multiprocessing for the stability fix

# Suppress common runtime warnings from gplearn (e.g., division by zero during evolution)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)

# This implementation assumes a DataFrame input and defines the target internally.
def evolve_features(df: pd.DataFrame, target: pd.Series, num_features: int = 5, **kwargs):
    """
    Evolves new features using Genetic Programming (gplearn.SymbolicTransformer).
    Optimized for speed and stability within the Forge architecture.
    """
    
    # --- 1. Data Preparation ---
    # The target is now passed directly to the function.
    df['target_return'] = target

    # Select numeric features for input
    feature_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Define columns that should not be used as inputs (to prevent leakage/redundancy)
    exclude_cols = ['target_return', 'open', 'high', 'low', 'close', 'volume']
    input_features = [col for col in feature_cols if col not in exclude_cols]

    if not input_features:
        return pd.DataFrame(index=df.index)

    # Clean data (gplearn requires no NaNs or Infs for fitting)
    df_clean = df[input_features + ['target_return']].replace([np.inf, -np.inf], np.nan).dropna()
    
    # Ensure sufficient data remains
    if df_clean.empty or len(df_clean) < 200:
        logger.warning("Not enough clean data for feature evolution. Skipping.")
        return pd.DataFrame(index=df.index)

    X = df_clean[input_features]
    y = df_clean['target_return']

    # --- 2. Stability Fix: Deadlock Prevention (CRITICAL) ---
    # When running inside a Forge worker (subprocess), nested parallelism causes deadlocks.
    # We must force n_jobs=1 in subprocesses.
    is_subprocess = multiprocessing.current_process().name != 'MainProcess'
    if is_subprocess:
        n_jobs = 1
        logger.debug("[Feature Synthesizer] Subprocess detected. Setting n_jobs=1.")
    else:
        # Use available cores if running standalone in the main process
        n_jobs = multiprocessing.cpu_count()
        logger.debug(f"[Feature Synthesizer] Main process detected. Setting n_jobs={n_jobs}.")
    # ---------------------------------------------

    # --- 3. Optimized GP Parameters (Faster Execution) ---
    # Balanced parameters for speed and exploration
    params = {
        'generations': 25,          # Reduced for speed
        'population_size': 1000,    # Balanced exploration
        'hall_of_fame': 100,
        'n_components': num_features,
        'metric': 'spearman',       # Spearman correlation is fast and robust
        'parsimony_coefficient': 0.005, # Favors simplicity, faster convergence
        'max_samples': 0.9,         # Internal sampling for robustness
        'verbose': 1,
        'random_state': 42,
        'n_jobs': n_jobs,           # Apply the safe n_jobs value
        'function_set': ('add', 'sub', 'mul', 'div', 'log', 'sqrt', 'abs', 'neg', 'inv', 'max', 'min')
    }
    params.update(kwargs) # Allow overrides if necessary

    # --- 4. Evolution Process (SymbolicTransformer) ---
    transformer = SymbolicTransformer(**params)

    try:
        logger.info(f"[Feature Synthesizer] Starting optimized evolution. Pop={params['population_size']}, Gen={params['generations']}...")
    # --- V9.5 PERFORMANCE FIX ---
    # Force the 'loky' backend for joblib. This is more robust than the default
    # and ensures that gplearn can use all available cores even when it's
    # running inside a subprocess (like a Forge Worker), fixing the low CPU usage issue.
    import joblib
    with joblib.parallel_backend('loky', n_jobs=n_jobs):
        est.fit(X, y)
    # --- END FIX ---
        
        # Transform the original (full) data using the evolved features
        # Use forward fill and then zero fill on the original inputs for transformation stability
        X_original_inputs = df[input_features].ffill().fillna(0) 
        
        # Handle potential errors during transformation
        try:
            X_transformed = transformer.transform(X_original_inputs)
        except Exception as transform_e:
            logger.error(f"Error during transformation on full dataset: {transform_e}. Returning empty.", exc_info=True)
            return pd.DataFrame(index=df.index)

        # Create the resulting DataFrame
        evolved_df = pd.DataFrame(X_transformed, index=df.index)
        evolved_df.columns = [f"evolved_feat_{i+1}" for i in range(X_transformed.shape[1])]
        
        # Final cleanup of generated features
        evolved_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Fill any remaining NaNs resulting from transformations (e.g., log(0))
        evolved_df.fillna(0, inplace=True) 

        logger.info(f"[Feature Synthesizer] Evolution complete. Generated {len(evolved_df.columns)} features.")
        return evolved_df

    except Exception as e:
        logger.error(f"Error during GP fitting: {e}", exc_info=True)
        return pd.DataFrame(index=df.index)