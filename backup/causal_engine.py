
# forge/modeling/causal_engine.py

import numpy as np
import pandas as pd
import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
import logging

logger = logging.getLogger(__name__)

def run_causal_discovery(df: pd.DataFrame, var_names: list, target_variable: str, tau_max: int = 3, pc_alpha: float = 0.05, alpha_level: float = 0.01):
    """
    Runs the PCMCI causal discovery algorithm to identify causal drivers of a target variable.

    Args:
        df (pd.DataFrame): DataFrame containing the time series data.
        var_names (list): List of variable names in the DataFrame.
        target_variable (str): The name of the target variable for which to find causal parents.
        tau_max (int): Maximum time lag to consider for causal links.
        pc_alpha (float): Significance level for the parent candidate selection (PC-stage).
        alpha_level (float): Significance level for thresholding the resulting p-value matrix.

    Returns:
        set: A set of causal parent variable names.
    """
    if df.empty or not var_names:
        logger.warning("[CDE] Input DataFrame or var_names is empty. Skipping causal discovery.")
        return set()

    # CRITICAL FIX: Filter out constant features
    variances = df[var_names].var(numeric_only=True)
    # Use a small threshold (1e-9) instead of exactly 0 for float comparison
    non_constant_features = variances[variances > 1e-9].index.tolist()
    
    if target_variable not in non_constant_features:
        logger.warning(f"[CDE] Target '{target_variable}' is constant or non-numeric. Skipping CDE.")
        return set()

    df_filtered = df[non_constant_features]
    var_names_filtered = non_constant_features

    # Log removed columns
    removed_count = len(var_names) - len(var_names_filtered)
    if removed_count > 0:
        logger.info(f"[CDE] Removed {removed_count} constant features.")

    if len(df_filtered.columns) < 2:
        return set()

    try:
        # Create a Tigramite DataFrame object
        dataframe = pp.DataFrame(df_filtered.values, var_names=var_names_filtered)

        # Initialize a conditional independence test (Partial Correlation for linear relationships)
        parcorr = ParCorr(significance='analytic')

        # Initialize the PCMCI object
        pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=parcorr,
            verbosity=0  # Set to 1 for more detailed output
        )

        logger.info(f"[CDE] Running causal discovery for target '{target_variable}' with tau_max={tau_max}...")

        # Run the PCMCI algorithm
        results = pcmci.run_pcmci(
            tau_max=tau_max,
            pc_alpha=pc_alpha,
            alpha_level=alpha_level
        )

        # Get the causal parents of the target variable
        causal_parents = get_causal_parents(results, var_names, target_variable)
        
        logger.info(f"[CDE] Found {len(causal_parents)} causal parents for '{target_variable}': {causal_parents}")
        
        return causal_parents

    except Exception as e:
        logger.error(f"[CDE] Error during causal discovery: {e}", exc_info=True)
        return set() # Return an empty set in case of an error

def get_causal_parents(results: dict, var_names: list, target_variable: str) -> set:
    """
    Extracts the causal parents of a target variable from the PCMCI results.

    Args:
        results (dict): The results dictionary from the PCMCI run.
        var_names (list): List of variable names.
        target_variable (str): The target variable.

    Returns:
        set: A set of causal parent variable names.
    """
    causal_parents = set()
    if target_variable not in var_names:
        return causal_parents

    target_idx = var_names.index(target_variable)
    graph = results['graph']

    # --- ROBUSTNESS FIX: Check if tau_max is in results ---
    if 'tau_max' not in results:
        logger.warning("[CDE] 'tau_max' not found in PCMCI results. Skipping parent extraction.")
        return causal_parents
    # ---------------------------------------------------------

    # The graph has shape (N, N, tau_max + 1)
    # graph[j, i, tau] is the link from var i (t-tau) to var j (t)
    for i in range(len(var_names)): # Source variable
        for tau in range(1, results['tau_max'] + 1): # Lag
            if graph[target_idx, i, tau]:
                causal_parents.add(var_names[i])
    
    return causal_parents
