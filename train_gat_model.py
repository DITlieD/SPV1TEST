"""
Offline Training Script for the Influence Mapper's GAT Model
"""
import os
import pandas as pd
import numpy as np
import torch
import logging
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader as TorchDataLoader

# Add project root to path
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import config as app_config
from data_fetcher import load_historical_data
from forge.modeling.macro_causality import InfluenceMapper, PermutationEntropyCalculator, TransferEntropyCalculator, calculate_permutation_entropy

# A lightweight pipeline for training purposes
class TrainingPipeline:
    def __init__(self, pe_calculator, te_calculator):
        self.pe_calculator = pe_calculator
        self.te_calculator = te_calculator
        self.influence_mapper = InfluenceMapper(pe_calculator, te_calculator)

logging.basicConfig(level=app_config.LOG_LEVEL, format='%(asctime)s - [GAT Trainer] - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainGAT")

def create_graph_snapshots(cross_asset_data: dict, asset_list: list, pipeline: TrainingPipeline) -> list:
    """
    Processes historical data to create a sequence of graph snapshots for training.
    """
    logger.info("Creating historical graph snapshots...")
    
    # Align all data into a single DataFrame
    aligned_df = pd.DataFrame()
    for symbol, df in cross_asset_data.items():
        aligned_df[f"{symbol}_close"] = df['close']
    aligned_df.ffill(inplace=True)
    aligned_df.dropna(inplace=True)

    graph_snapshots = []
    # We need a lookback window for PE/TE, and a forward window for the target
    for i in range(pipeline.te_calculator.window_size, len(aligned_df) - 1):
        window_df = aligned_df.iloc[i - pipeline.te_calculator.window_size : i]
        
        pe_values = {}
        returns = {}
        volatility = {}

        for asset in asset_list:
            asset_close = window_df[f"{asset}_close"]
            asset_returns = asset_close.pct_change().dropna()
            
            pe = calculate_permutation_entropy(asset_returns.values)
            pe_values[asset] = pe if not np.isnan(pe) else 0.5

            returns[asset] = asset_returns.iloc[-1] if not asset_returns.empty else 0.0
            
            vol = asset_returns.std()
            volatility[asset] = vol if not np.isnan(vol) else 0.0

        # Calculate TE matrix for the window
        te_matrix = pipeline.te_calculator.calculate_causal_matrix(asset_list=asset_list, parallel=False)

        # Create the graph for the current state (t)
        graph_t = pipeline.influence_mapper.create_graph_snapshot(asset_list, pe_values, returns, volatility, te_matrix)

        # Calculate the target: future change in entropy for each asset
        future_pe = {}
        for asset in asset_list:
            future_series = aligned_df[f"{asset}_close"].iloc[i+1 : i+1 + pipeline.pe_calculator.window_size].pct_change().dropna()
            if not future_series.empty:
                fpe = calculate_permutation_entropy(future_series.values)
                future_pe[asset] = fpe if not np.isnan(fpe) else pe_values[asset]
            else:
                future_pe[asset] = pe_values[asset]
        
        delta_values = [future_pe[asset] - pe_values[asset] for asset in asset_list]
        delta_values = np.nan_to_num(delta_values)
        entropy_delta = torch.tensor(delta_values, dtype=torch.float).unsqueeze(1)
        graph_t.y = entropy_delta
        
        graph_snapshots.append(graph_t)

    logger.info(f"Created {len(graph_snapshots)} graph snapshots.")
    return graph_snapshots

def main():
    logger.info("=== Starting Influence Mapper GAT Model Training Pipeline ===")
    
    # 1. Load Data
    logger.info("--- Phase 1: Loading Historical Data ---")
    cross_asset_data = {}
    asset_list_sanitized = [s.replace('/', '').split(':')[0] for s in app_config.ASSET_UNIVERSE]
    
    for symbol in app_config.ASSET_UNIVERSE:
        df = load_historical_data(symbol, app_config.TIMEFRAMES['strategic'], limit=5000)
        if not df.empty:
            cross_asset_data[symbol] = df

    if not cross_asset_data:
        logger.error("No historical data found. Aborting.")
        return

    # 2. Initialize Mapper and Create Snapshots
    logger.info("--- Phase 2: Creating Graph Snapshots ---")
    pe_calc = PermutationEntropyCalculator(window_size=app_config.ACN_CONFIG['im_entropy_window'])
    te_calc = TransferEntropyCalculator(window_size=app_config.ACN_CONFIG['im_entropy_window'])
    # The mapper here is actually the pipeline
    mapper_pipeline = TrainingPipeline(pe_calc, te_calc)

    snapshots = create_graph_snapshots(cross_asset_data, app_config.ASSET_UNIVERSE, mapper_pipeline)

    if not snapshots:
        logger.error("Failed to create graph snapshots. Aborting.")
        return

    # 3. Train GAT Model
    logger.info("--- Phase 3: Training GAT Model ---")
    mapper_pipeline.influence_mapper.train_gat(snapshots, epochs=50, batch_size=32) # epochs can be tuned

    # 4. Save Model
    logger.info("--- Phase 4: Saving Model ---")
    os.makedirs(app_config.MODEL_DIR, exist_ok=True)
    model_path = os.path.join(app_config.MODEL_DIR, "influence_gat_model.pth")
    torch.save(mapper_pipeline.influence_mapper.gat_model.state_dict(), model_path)
    logger.info(f"GAT model saved successfully to: {model_path}")
    logger.info("=== Pipeline Complete ===")

if __name__ == "__main__":
    main()
