# train_sde_brain.py
import pandas as pd
import numpy as np
import logging
import os
import time
import config as app_config

# Import Data Handlers
from data_fetcher import load_historical_data
from forge.data_processing.l2_collector import load_historical_l2_data

# Assuming FeatureFactory and labeling are implemented correctly
try:
    from forge.data_processing.feature_factory import FeatureFactory
except ImportError:
     print("WARNING: FeatureFactory not found. Standard feature engineering will be skipped.")
     FeatureFactory = None

# Assuming labeling.py exists
try:
    from forge.data_processing.labeling import create_categorical_labels
except ImportError:
     print("WARNING: create_categorical_labels not found. Using placeholder.")
     def create_categorical_labels(df):
        return pd.Series(np.random.randint(0, 3, size=len(df)), index=df.index)

# Import ACN Components
from forge.data_processing.microstructure_features_l2 import calculate_microstructure_features
from forge.modeling.micro_causality import run_mce_inference
from forge.modeling.macro_causality import calculate_influence_map
from forge.evolution.symbiotic_crucible import ImplicitBrain

logging.basicConfig(level=app_config.LOG_LEVEL, format='%(asctime)s - [SDE Trainer] - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainSDEBrain")

TRAINING_WINDOW_SIZE = app_config.ACN_CONFIG.get('training_window', 10000)
LTF = app_config.TIMEFRAMES['microstructure']
MODEL_FRESHNESS_HOURS = 168 # 7 days

def main():
    logger.info(f"=== Starting SDE Implicit Brain Training Pipeline (ACN) ===")

    for target_asset in app_config.ASSET_UNIVERSE:
        logger.info(f"\n{'='*60}\n--- Processing Asset: {target_asset} ---\n{'='*60}")

        # 1. Model Freshness Check
        model_name = target_asset.replace('/', '').replace(':', '_')
        model_path = os.path.join(app_config.MODEL_DIR, f"sde_implicit_brain_{model_name}.pkl")

        if os.path.exists(model_path):
            model_age_hours = (time.time() - os.path.getmtime(model_path)) / 3600
            if model_age_hours < MODEL_FRESHNESS_HOURS:
                logger.info(f"Implicit Brain for {target_asset} is recent ({model_age_hours:.1f}h old). Skipping.")
                continue
            else:
                logger.info(f"Implicit Brain for {target_asset} is outdated ({model_age_hours:.1f}h old). Retraining...")

        # 2. Load Cross-Asset Data
        logger.info("--- Phase 1: Data Loading (Universe and L2) ---")
        cross_asset_data = {}
        for symbol in app_config.ASSET_UNIVERSE:
            df = load_historical_data(symbol, LTF, limit=TRAINING_WINDOW_SIZE)
            if not df.empty:
                cross_asset_data[symbol] = df

        if target_asset not in cross_asset_data:
            logger.error(f"Target asset L1 data for {target_asset} missing. Aborting this asset.")
            continue

        df_l2 = load_historical_l2_data(target_asset)

        # 3. Standard Feature Engineering
        logger.info("--- Phase 2: Standard Feature Engineering (FeatureFactory) ---")
        if FeatureFactory:
            ff = FeatureFactory(cross_asset_data.copy(), logger=logger)
            processed_data = ff.create_all_features(app_config, exchange=None)
            df_features = processed_data[target_asset]
        else:
            df_features = cross_asset_data[target_asset].copy()

        # 4. MCE Integration
        logger.info("--- Phase 3: MCE Integration ---")
        if not df_l2.empty:
            df_microstructure = calculate_microstructure_features(df_l2)
            df_mce_signals = run_mce_inference(df_microstructure)
            logger.info("Aligning MCE features onto LTF using merge_asof...")
            df_features = pd.merge_asof(df_features.sort_index(), df_mce_signals.sort_index(), left_index=True, right_index=True, direction='backward')
        else:
            logger.warning("L2 data unavailable. Proceeding without MCE features.")

        # 5. Influence Mapper Integration
        logger.info("--- Phase 4: Influence Mapper Integration ---")
        df_influence = calculate_influence_map(cross_asset_data, target_asset, app_config.ACN_CONFIG)
        if not df_influence.empty:
            logger.info("Aligning Influence Mapper features using merge_asof...")
            df_features = pd.merge_asof(df_features.sort_index(), df_influence.sort_index(), left_index=True, right_index=True, direction='backward')
        else:
             logger.warning("Influence Map calculation failed. Proceeding without Macro features.")

        # 6. Labeling and Sanitization
        logger.info("--- Phase 5: Labeling and Sanitization ---")
        if 'close' not in df_features.columns:
             logger.error("'close' price missing for labeling. Aborting.")
             continue

        df_features['label'] = create_categorical_labels(df_features)
        df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_features.dropna(inplace=True)

        if df_features.empty:
            logger.error("Dataframe empty after sanitization. Aborting asset.")
            continue

        # 7. Implicit Brain Training
        logger.info("--- Phase 6: Implicit Brain Training (SDE) ---")
        feature_columns = [col for col in df_features.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'label']]
        X = df_features[feature_columns]
        y = df_features['label']

        brain = ImplicitBrain(feature_list=feature_columns)
        success = brain.train(X, y)

        if success:
            brain.save_model(model_path)
            logger.info(f"=== Pipeline for {target_asset} Complete Successfully ===")
        else:
            logger.error(f"=== Pipeline for {target_asset} Failed ===")

if __name__ == "__main__":
    os.makedirs(app_config.MODEL_DIR, exist_ok=True)
    os.makedirs(app_config.DATA_DIR, exist_ok=True)
    main()
