"""
Offline Training Script for MCEFeatureNormalizer
"""
import os
import pandas as pd
import joblib
import logging

from config import MODEL_DIR, DATA_DIR
from forge.data_processing.l2_collector import load_historical_l2_data
from forge.data_processing.microstructure_features_l2 import calculate_microstructure_features
from forge.modeling.micro_causality import MCEFeatureNormalizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MCENormalizerTrainer")

def main():
    logger.info("--- Starting MCEFeatureNormalizer Training ---")

    # 1. Load historical L2 data for a representative asset
    target_asset = "BTC/USDT:USDT"
    logger.info(f"Loading historical L2 data for {target_asset}...")
    df_l2 = load_historical_l2_data(target_asset)

    if df_l2.empty:
        logger.error("No L2 data found. Cannot train the normalizer. Please ensure L2 snapshot data exists.")
        return

    # 2. Calculate historical microstructure features
    logger.info("Calculating historical microstructure features...")
    df_microstructure = calculate_microstructure_features(df_l2)

    if df_microstructure.empty:
        logger.error("Feature calculation resulted in an empty DataFrame. Aborting.")
        return

    # 3. Fit the normalizer
    logger.info("Fitting the MCEFeatureNormalizer...")
    normalizer = MCEFeatureNormalizer()
    
    # The normalizer's `normalize_features` method updates its internal stats,
    # so we can "fit" it by running it over the historical data.
    for i in range(len(df_microstructure)):
        normalizer.normalize_features(df_microstructure.iloc[i])

    logger.info("Normalizer fitting complete.")

    # 4. Save the fitted normalizer
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, "mce_normalizer.pkl")
    joblib.dump(normalizer, save_path)
    logger.info(f"MCEFeatureNormalizer saved successfully to: {save_path}")
    logger.info("--- Training Complete ---")

if __name__ == "__main__":
    main()
