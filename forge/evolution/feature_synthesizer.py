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
import joblib
import tempfile
import shutil

# Suppress common runtime warnings from gplearn (e.g., division by zero during evolution)
def evolve_features(df: pd.DataFrame, target: pd.Series, num_features: int = 10, population_size: int = 100, generations: int = 5, n_jobs: int = -1) -> pd.DataFrame:
    """
    Evolves new features from a given DataFrame using genetic programming.
    """
    logger = logging.getLogger(__name__)
    
    # --- 1. Parameter Setup ---
    params = {
        'population_size': population_size,
        'generations': generations,
        'function_set': ('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv'),
        'n_components': num_features,
        'metric': 'pearson',
        'stopping_criteria': 0.99,
        'p_crossover': 0.9,
        'p_subtree_mutation': 0.01,
        'p_hoist_mutation': 0.01,
        'p_point_mutation': 0.01,
        'max_samples': 0.9,
        'verbose': 0,
        'random_state': 0,
        'n_jobs': 1 # Use 1 job internally for gplearn as we parallelize with joblib
    }

    # --- 2. Data Preparation ---
    input_features = [col for col in df.columns if df[col].dtype in [np.float64, np.int64]]
    X = df[input_features].copy()
    y = target.copy()

    # Align X and y
    aligned_index = X.index.intersection(y.index)
    X = X.loc[aligned_index]
    y = y.loc[aligned_index]

    # Handle potential NaN/Inf values
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(method='ffill', inplace=True)
    X.fillna(method='bfill', inplace=True)
    X.fillna(0, inplace=True)

    if X.empty:
        logger.warning("[Feature Synthesizer] Input DataFrame is empty after cleaning. Cannot evolve features.")
        return pd.DataFrame(index=df.index)

    # --- 3. Set Multiprocessing Start Method (if not already set) ---
    # This is a common requirement for cross-platform stability
    try:
        if multiprocessing.get_start_method(allow_none=True) is None:
            multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # Already set, which is fine.
        pass

    # --- 4. Evolution Process (SymbolicTransformer) ---
    transformer = SymbolicTransformer(**params)

    # --- V9.7 STABILITY FIX for Nested Parallelism ---
    temp_folder = tempfile.mkdtemp()
    try:
        logger.info(f"[Feature Synthesizer] Starting optimized evolution. Pop={params['population_size']}, Gen={params['generations']}...")
        with joblib.parallel_backend('loky', n_jobs=n_jobs, temp_folder=temp_folder):
            transformer.fit(X, y)
        
        # Transform the original (full) data using the evolved features
        X_original_inputs = df[input_features].ffill().fillna(0) 
        
        try:
            X_transformed = transformer.transform(X_original_inputs)
        except Exception as transform_e:
            logger.error(f"Error during transformation on full dataset: {transform_e}. Returning empty.", exc_info=True)
            return pd.DataFrame(index=df.index)

        # Create the resulting DataFrame
        evolved_df = pd.DataFrame(X_transformed, index=df.index)
        evolved_df.columns = [f"evolved_feat_{i+1}" for i in range(X_transformed.shape[1])]
        
        evolved_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        evolved_df.fillna(0, inplace=True) 

        logger.info(f"[Feature Synthesizer] Evolution complete. Generated {len(evolved_df.columns)} features.")
        return evolved_df

    except Exception as e:
        logger.error(f"Error during GP fitting: {e}", exc_info=True)
        return pd.DataFrame(index=df.index)
    finally:
        shutil.rmtree(temp_folder, ignore_errors=True)