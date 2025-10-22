# feature_analyzer.py
"""
Cross-Asset Feature Importance Analyzer and Pruner

**Purpose:**
This script automates the process of feature selection by identifying and removing
features that provide little to no value across the entire asset universe. By
systematically pruning these features, we can reduce model complexity, decrease
training time, and potentially improve the generalization of our trading models.

**Functionality:**
1.  **Iterate Assets:** The script loops through every asset defined in `config.ASSET_UNIVERSE`.
2.  **Data Preparation:** For each asset, it fetches a large historical dataset and runs the complete
    feature engineering pipeline.
3.  **Model Training:** A baseline `SimpleConfluenceModel` is trained on the first part of the
    historical data.
4.  **OOS Importance Calculation:** Using the trained model and the out-of-sample (unseen) data,
    it calculates the Permutation Feature Importance (PFI). PFI measures how much the model's
    performance degrades when a single feature's values are shuffled, providing a robust
    metric for feature relevance.
5.  **Aggregation:** It collects the PFI results from all assets and calculates the average
    importance score for each feature across the entire universe.
6.  **Pruning:** It identifies all features with an average importance score that is less than
    or equal to zero. These are considered "useless" or "harmful".
7.  **Automatic Configuration Update:** The script then reads the `config.py` file, finds the lines
    in the `CORE_FEATURES` and `PHASE1_FEATURES` lists that contain the useless features,
    and removes them.

**Limitations and Future Improvements:**
-   **Simple Pruning Logic:** The current logic removes a feature if its *average* importance is
    non-positive. A more advanced approach could consider the distribution of scores, such as
    removing a feature only if it's non-positive for >70% of the assets.
-   **Hardcoded Threshold:** The pruning threshold is fixed at `<= 0`. This could be made a
    configurable parameter to allow for more or less aggressive pruning.
-   **Single Model Analysis:** The analysis is based on the `SimpleConfluenceModel`. Feature
    importance can vary between different model architectures. A more comprehensive analysis
    could aggregate results from multiple model types.
-   **Direct File Modification:** The script directly overwrites `config.py`. A safer version
    might create a new `config.py.new` file and require manual confirmation before replacing
    the original.
"""

import os
import pandas as pd
from config import ASSET_UNIVERSE, CORE_FEATURES, PHASE1_FEATURES
from elite_tuner import get_oos_permutation_importance
from models_v2 import SimpleConfluenceModel
from data_processing_v2 import get_market_data, add_all_features, get_specialist_labels

def analyze_feature_importance_across_assets():
    """
    Performs a cross-asset feature importance analysis and returns a DataFrame
    of aggregated feature importances.
    """
    all_importances = []
    for asset in ASSET_UNIVERSE:
        print(f"\n--- Analyzing {asset} ---")
        
        # 1. Prepare data
        df_full = get_market_data(asset, '4h', limit=10000)
        if df_full.empty:
            continue
            
        df_features = add_all_features(df_full.copy())
        atr_col = next((c for c in df_features.columns if c.startswith('ATRr')), 'ATRr_14')
        df_features['label'] = get_specialist_labels(df_features, atr_col, module_type='reversion')
        df_features.dropna(inplace=True)

        # 2. Train a simple model
        model = SimpleConfluenceModel()
        feature_set = model.select_features(df_features)
        df_train, df_test = df_features.iloc[:-1000], df_features.iloc[-1000:]
        model.fit(df_train)

        # 3. Get OOS permutation importance
        pfi_df = get_oos_permutation_importance(model, df_test, feature_set)
        pfi_df['asset'] = asset
        all_importances.append(pfi_df)

    if not all_importances:
        print("No feature importances were generated.")
        return pd.DataFrame()

    # 4. Aggregate results
    aggregated_df = pd.concat(all_importances)
    feature_summary = aggregated_df.groupby('feature')['importance_mean'].mean().sort_values(ascending=True)
    
    return feature_summary

def prune_features_in_config(features_to_remove):
    """
    Removes the given features from the feature lists in config.py.
    """
    with open('config.py', 'r') as f:
        lines = f.readlines()

    def is_feature_line(line):
        return any(f"'{feature}'" in line for feature in features_to_remove)

    new_lines = [line for line in lines if not is_feature_line(line)]

    with open('config.py', 'w') as f:
        f.writelines(new_lines)

if __name__ == '__main__':
    print("--- Starting Cross-Asset Feature Importance Analysis ---")
    feature_summary = analyze_feature_importance_across_assets()
    
    if not feature_summary.empty:
        print("\n--- Aggregated Feature Importances ---")
        print(feature_summary)
        
        # Identify features with negative importance
        useless_features = feature_summary[feature_summary <= 0].index.tolist()
        
        if useless_features:
            print(f"\n--- Pruning {len(useless_features)} features with non-positive importance ---")
            print(useless_features)
            prune_features_in_config(useless_features)
            print("\n--- config.py has been updated. ---")
        else:
            print("\n--- No features to prune. ---")
